import numpy as np
import SimpleITK
import pathlib
import time

from neuroglancer_interface.utils.rotation_utils import (
    get_rotation_matrix,
    rotate_matrix)


class NiftiArray(object):
    """
    A class to carry around and self-consistently manipulate
    the image data from a NiftiArray and its geometric metadata
    """

    def __init__(self, nifti_path, transposition):
        if transposition is not None:
            raise NotImplementedError(
                "cannot support non None transposition")

        self.nifti_path = pathlib.Path(nifti_path)

        #_raw = (0, 1, 2)
        #_raw_rot = np.array([[1, 0, 0],[0,1,0],[0,0,1]])
        (_raw,
         _raw_rot) = self._get_raw_transposition()

        if transposition is None:
            self.transposition = _raw
            self.rotation_matrix = _raw_rot
        else:
            self.transposition = (_raw[transposition[0]],
                                  _raw[transposition[1]],
                                  _raw[transposition[2]])

        self.img_transposition = (self.transposition[2],
                                  self.transposition[1],
                                  self.transposition[0])

        #print(f"img_transposition {self.img_transposition}")

        if not self.nifti_path.is_file():
            raise RuntimError(f"{self.nifti_path} is not a file")

    def _read_quatern_terms(self):
        img = SimpleITK.ReadImage(self.nifti_path)
        self._quatern_b = float(img.GetMetaData('quatern_b'))
        self._quatern_c = float(img.GetMetaData('quatern_c'))
        self._quatern_d = float(img.GetMetaData('quatern_d'))
        qsq = self._quatern_b**2+self._quatern_c**2+self._quatern_d**2
        if qsq > 1.0:
            self._quatern_a = 0.0
        else:
            self._quatern_a = np.sqrt(1.0-qsq)

    def _get_raw_transposition(self):
        """
        Convert the quaternion terms from the NIFTI header into
        an image matrix. Determine what this means about the
        way to transpose x, y, z. Apply this transposition to
        our default _raw transposition; return that _raw
        transposition.

        See:
        https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/quatern.html
        """
        self._read_quatern_terms()

        rotation_matrix = get_rotation_matrix(
            aa = self._quatern_a,
            bb = self._quatern_b,
            cc = self._quatern_c,
            dd = self._quatern_d)

        bases = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])

        mapping = dict()
        been_chosen = set()
        for i_orig in range(3):
            v_orig = bases[i_orig, :]
            v_new = np.dot(rotation_matrix, v_orig)
            chosen = None
            for i_other in range(3):
                if np.allclose(v_new,
                               bases[i_other, :],
                               rtol=1.0e-5,
                               atol=1.0e-5):
                    chosen = i_other
                    break

                if np.allclose(v_new,
                               -1.0*bases[i_other, :],
                               rtol=1.0e-5,
                               atol=1.0e-5):
                    chosen = i_other
                    break

            if chosen is None:
                raise RuntimeError(
                    f"quaternion terms\n{aa:.5f}\n{bb:.5f}\n"
                    f"{cc:.5f}\n{dd:.5f}\n"
                    f"do not neatly map bases onto each other\n"
                    f"orig {v_orig}\n"
                    f"new {v_new}")
            mapping[i_orig] = chosen
            assert chosen not in been_chosen
            been_chosen.add(chosen)

        _raw = (mapping[0],
                mapping[1],
                mapping[2])

        return _raw, rotation_matrix


    def _read_metadata(self):
        t0 = time.time()
        print('reading image')
        img = SimpleITK.ReadImage(self.nifti_path)
        print(f'reading took {time.time()-t0:.2e} seconds')
        _raw_shape = img.GetSize()
        _raw_shape = np.array([_raw_shape[2],
                               _raw_shape[1],
                               _raw_shape[0]])
        print(f"shape before rot {_raw_shape}")
        self._shape = tuple(np.abs(
                               np.round(
                                   np.dot(self.rotation_matrix,
                                          _raw_shape))).astype(int))

        self._shape = tuple([int(self._shape[idx]) for idx in range(3)])

        self._scales = self._get_scales(img)
        print(f"assigned shape {self._shape}")
        print(f"assigned scales {self._scales}")
        self._img = img

    @property
    def shape(self):
        if not hasattr(self, '_shape'):
            self._read_metadata()
        return self._shape

    @property
    def arr(self):
        if not hasattr(self, '_arr'):
            self._arr = self._get_arr()
            if not self._arr.shape[:3] == self.shape:
                raise RuntimeError(f"arr shape {self._arr.shape};\n"
                                   f"should be {self.shape}")
        return self._arr

    @property
    def scales(self):
        if not hasattr(self, '_scales'):
            self._read_metadata()
        return self._scales


    def _get_arr(self) -> np.ndarray:
        """
        Will be cast so that arr.shape matches img.GetSize()
        """
        print("getting array")
        expected_shape = self.shape  # just to provoke metadata read
        img = self._img
        arr = SimpleITK.GetArrayFromImage(img)
        print(f"raw arr shape {arr.shape}")
        if len(arr.shape) == 3:
            return rotate_matrix(arr, self.rotation_matrix)
        elif len(arr.shape) == 4:
            return np.stack([rotate_matrix(arr[:, :, :, ix],
                                            self.rotation_matrix)
                             for ix in range(arr.shape[3])]).transpose(1,2,3,0)
        else:
            raise RuntimeError(
                f"Cannot parse array of shape {arr.shape}")

    def _get_scales(self, img) -> tuple:
        """
        Returns dimensions in (1, 2, 3) order as they appear
        in the NIFTI file
        """
        print("reading scales")
        d1_mm = img.GetMetaData('pixdim[1]')
        d2_mm = img.GetMetaData('pixdim[2]')
        d3_mm = img.GetMetaData('pixdim[3]')
        _raw = np.array([float(d3_mm),
                         float(d2_mm),
                         float(d1_mm)])

        print(f"raws scales {_raw}")
        print(f"raw shape {img.GetSize()}")
        print(f"rotation {self.rotation_matrix}")
        return tuple(np.abs(np.dot(self.rotation_matrix, _raw)))

    def get_channel(self, channel):

        if channel not in ('green', 'red', 'blue', None):
            raise RuntimeError(
                f"invalid channel: {channel}")

        if channel is None:
            channel_idx = 0
        else:
            channel_idx = {'red': 0, 'green': 1, 'blue': 2}[channel]

        if len(self.arr.shape) == 4:
            this_channel = self.arr[:, :, :, channel_idx]
        else:
            this_channel = self.arr

        return {'channel': this_channel,
                'scales': self.scales}


class NiftiArrayCollection(object):

    def __init__(self, nifti_dir_path, transposition=None):
        print("in dir path constructor")
        nifti_dir_path = pathlib.Path(nifti_dir_path)
        if not nifti_dir_path.is_dir():
            raise RuntimeError(
                f"{nifti_dir_path} is not a dir")

        self.transposition = transposition
        print(f"getting path list {nifti_dir_path}")
        path_list = [n for n in nifti_dir_path.rglob('*.nii.gz')]
        print(path_list)
        channel_lookup = dict()
        for path in path_list:
            if 'green' in path.name:
                key = 'green'
            elif 'red' in path.name:
                key = 'red'
            elif 'blue' in path.name:
                key = 'blue'
            else:
                continue

            if key in channel_lookup:
                msg = f"More than one path for channel: {key}\n"
                msg += f"{path.resolve().absolute()}\n"
                msg += f"{channel_lookup[key].resolve().absolute()}"
                raise RuntimeError(msg)

            channel_lookup[key] = path
        self.channel_lookup = channel_lookup

    @property
    def scales(self):
        if not hasattr(self, '_scales'):
            k = list(self.channel_lookup.keys())[0]
            this = NiftiArray(self.channel_lookup[k],
                              transposition=self.transposition)
            self._scales = this.scales
        return self._scales

    @property
    def shape(self):
        if not hasattr(self, '_shape'):
            print("reading shape")
            self._shape = None
            for k in self.channel_lookup:
                this = NiftiArray(self.channel_lookup[k],
                                  transposition=self.transposition)
                if self._shape is None:
                    self._shape = this.shape
                else:
                    assert this.shape == self._shape
        return self._shape

    def get_channel(self, channel):
        if channel is None:
            channel = 'red'
        this_path = self.channel_lookup[channel]
        nifti_array = NiftiArray(this_path, transposition=self.transposition)

        return nifti_array.get_channel(
                    channel=None)


def get_nifti_obj(nifti_path, transposition=None):
    nifti_path = pathlib.Path(nifti_path)
    if nifti_path.is_dir():
        print("getting NiftiARrayCollection")
        return NiftiArrayCollection(nifti_path, transposition=transposition)
    elif nifti_path.is_file():
        print("getting NiftiArray")
        return NiftiArray(nifti_path, transposition=transposition)

    raise RuntimeError(
        f"{nifti_path} is neither file nor dir")
