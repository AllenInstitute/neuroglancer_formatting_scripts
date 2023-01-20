import numpy as np
import SimpleITK
import pathlib
import time


class NiftiArray(object):
    """
    A class to carry around and self-consistently manipulate
    the image data from a NiftiArray and its geometric metadata
    """

    def __init__(self, nifti_path, transposition):
        self.nifti_path = pathlib.Path(nifti_path)

        _raw = self._get_raw_transposition()

        if transposition is None:
            self.transposition = _raw
            self.img_transposition = (0, 1, 2)
        else:
            self.transposition = (_raw[transposition[0]],
                                  _raw[transposition[1]],
                                  _raw[transposition[2]])
            self.img_transposition = tuple(transposition)

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

        _raw = (2, 1, 0)  # because of difference in NIFTI and numpy conventions

        aa = self._quatern_a
        bb = self._quatern_b
        cc = self._quatern_c
        dd = self._quatern_d

        rot = np.zeros((3,3), dtype=float)
        rot[0, 0] = aa**2+bb**2-cc**2-dd**2
        rot[1, 1] = aa**2+cc**2-bb**2-dd**2
        rot[2, 2] = aa**2+dd**2-bb**2-cc**2
        rot[0, 1] = 2*bb*cc-2*aa*dd
        rot[0, 2] = 2*bb*dd+2*aa*cc
        rot[1, 0] = 2*bb*cc+2*aa*dd
        rot[1, 2] = 2*cc*dd-2*aa*bb
        rot[2, 0] = 2*bb*dd-2*aa*cc
        rot[2, 1] = 2*cc*dd+2*aa*bb

        bases = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])

        mapping = dict()
        been_chosen = set()
        for i_orig in range(3):
            v_orig = bases[i_orig, :]
            v_new = np.dot(rot, v_orig)
            chosen = None
            for i_other in range(3):
                if np.allclose(v_new,
                               bases[i_other, :],
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

        _raw = (mapping[_raw[0]],
                mapping[_raw[1]],
                mapping[_raw[2]])

        return _raw


    def _read_metadata(self):
        t0 = time.time()
        print('reading image')
        img = SimpleITK.ReadImage(self.nifti_path)
        print(f'reading took {time.time()-t0:.2e} seconds')
        _raw = img.GetSize()
        self._shape = (_raw[self.img_transposition[0]],
                       _raw[self.img_transposition[1]],
                       _raw[self.img_transposition[2]])
        self._scales = self._get_scales(img)
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
            assert self._arr.shape[:3] == self.shape
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
        if len(arr.shape) == 3:
            return arr.transpose(self.transposition)
        elif len(arr.shape) == 4:
            return arr.transpose(self.transposition[0],
                                 self.transposition[1],
                                 self.transposition[2],
                                 3)

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
        _raw = (float(d1_mm),
                float(d2_mm),
                float(d3_mm))
        return (_raw[self.transposition[0]],
                _raw[self.transposition[1]],
                _raw[self.transposition[2]])

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
