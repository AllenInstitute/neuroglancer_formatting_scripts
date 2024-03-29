import numpy as np
import SimpleITK
import pathlib
import time

from neuroglancer_interface.utils.rotation_utils import (
    rotate_matrix,
    get_rotation_matrix)


class NiftiArray(object):
    """
    A class to carry around and self-consistently manipulate
    the image data from a NiftiArray and its geometric metadata

    If do_transposition, then transpose the NIFTI volume such
    that (x, y, z) -> (z, y, x), mimicking the transposition
    between SimpleITK.Image.GetSize() and array.shape

    posxposz = True will change the transposiiton rotation
    matrix so that +x < - > +z (instead of +x -> +z, +z -> -x)
    """

    def __init__(
            self,
            nifti_path,
            do_transposition=False,
            posxposz=False):

        self._do_transposition = do_transposition
        self._posxposz = posxposz
        self.nifti_path = pathlib.Path(nifti_path)

        if not self.nifti_path.is_file():
            raise RuntimError(f"{self.nifti_path} is not a file")

    @property
    def do_transposition(self):
        return self._do_transposition

    @property
    def rotation_matrix(self):
        if not hasattr(self, '_rotation_matrix'):
            #self._rotation_matrix = self._get_rotation_matrix()
            if self.do_transposition:
                if self._posxposz:
                    self._rotation_matrix = np.array(
                        [[0.0, 0.0, 1.0],
                         [0.0, 1.0, 0.0],
                         [1.0, 0.0, 0.0]])
                else:
                    self._rotation_matrix = np.array(
                        [[0.0, 0.0, -1.0],
                         [0.0, 1.0, 0.0],
                         [1.0, 0.0, 0.0]])
            else:
                self._rotation_matrix = np.array(
                                [[1.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0],
                                 [0.0, 0.0, 1.0]])
        return self._rotation_matrix

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

    def _get_rotation_matrix(self):
        """
        Convert the quaternion terms from the NIFTI header into
        a rotation  matrix.

        See:
        https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/quatern.html
        """
        raise RuntimeError("should not be here")
        self._read_quatern_terms()
        rotation_matrix = get_rotation_matrix(
            aa = self._quatern_a,
            bb = self._quatern_b,
            cc = self._quatern_c,
            dd = self._quatern_d)

        return rotation_matrix


    def _read_metadata(self):
        t0 = time.time()
        img = SimpleITK.ReadImage(self.nifti_path)
        _raw_shape = img.GetSize()
        _raw_shape = np.array([_raw_shape[2],
                               _raw_shape[1],
                               _raw_shape[0]])
        self._shape = tuple(np.abs(
                               np.round(
                                   np.dot(self.rotation_matrix,
                                          _raw_shape))).astype(int))

        self._shape = tuple([int(self._shape[idx]) for idx in range(3)])

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
        needs_rotation = True
        if np.allclose(self.rotation_matrix,
                           np.identity(3),
                           atol=0.0,
                           rtol=1.0e-5):
            needs_rotation = False
        expected_shape = self.shape  # just to provoke metadata read
        img = self._img
        arr = SimpleITK.GetArrayFromImage(img).astype(np.float32)
        if len(arr.shape) == 3:
            if needs_rotation:
                return rotate_matrix(arr, self.rotation_matrix)
            else:
                return arr
        elif len(arr.shape) == 4:
            if needs_rotation:
                return np.stack([rotate_matrix(arr[:, :, :, ix],
                                                self.rotation_matrix)
                                 for ix in range(arr.shape[3])]).transpose(1,2,3,0)
            else:
                return np.stack([
                    arr[:, :, :, ix]
                    for ix in range(arr.shape[3])]).transpose(1,2,3,0)
        else:
            raise RuntimeError(
                f"Cannot parse array of shape {arr.shape}")

    def _get_scales(self, img) -> tuple:
        """
        Returns dimensions in (1, 2, 3) order as they appear
        in the NIFTI file
        """
        d1_mm = img.GetMetaData('pixdim[1]')
        d2_mm = img.GetMetaData('pixdim[2]')
        d3_mm = img.GetMetaData('pixdim[3]')
        _raw = np.array([float(d3_mm),
                         float(d2_mm),
                         float(d1_mm)])

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

    def __init__(
            self,
            nifti_dir_path,
            do_transposition=False,
            posxposz=False):
        self._do_transposition = do_transposition
        self._posxposz = posxposz
        nifti_dir_path = pathlib.Path(nifti_dir_path)
        if not nifti_dir_path.is_dir():
            raise RuntimeError(
                f"{nifti_dir_path} is not a dir")

        path_list = [n for n in nifti_dir_path.rglob('*.nii.gz')]
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
    def do_transposition(self):
        return self._do_transposition

    @property
    def scales(self):
        if not hasattr(self, '_scales'):
            k = list(self.channel_lookup.keys())[0]
            this = NiftiArray(self.channel_lookup[k],
                              do_transposition=self.do_transposition,
                              posxposz=self._posxposz)
            self._scales = this.scales
        return self._scales

    @property
    def shape(self):
        if not hasattr(self, '_shape'):
            self._shape = None
            for k in self.channel_lookup:
                this = NiftiArray(self.channel_lookup[k],
                                  do_transposition=self.do_transposition,
                                  posxposz=self._posxposz)
                if self._shape is None:
                    self._shape = this.shape
                else:
                    assert this.shape == self._shape
        return self._shape

    def get_channel(self, channel):
        if channel is None:
            channel = 'red'
        this_path = self.channel_lookup[channel]
        nifti_array = NiftiArray(
                        this_path,
                        do_transposition=self.do_transposition,
                        posxposz=self._posxposz)

        return nifti_array.get_channel(
                    channel=None)


def get_nifti_obj(nifti_path, do_transposition=False, posxposz=False):
    nifti_path = pathlib.Path(nifti_path)
    if nifti_path.is_dir():
        return NiftiArrayCollection(nifti_path,
                                    do_transposition=do_transposition,
                                    posxposz=posxposz)
    elif nifti_path.is_file():
        return NiftiArray(nifti_path,
                          do_transposition=do_transposition,
                          posxposz=posxposz)

    raise RuntimeError(
        f"{nifti_path} is neither file nor dir")
