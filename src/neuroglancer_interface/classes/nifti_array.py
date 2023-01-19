import numpy as np
import SimpleITK
import pathlib


class NiftiArray(object):
    """
    A class to carry around and self-consistently manipulate
    the image data from a NiftiArray and its geometric metadata
    """

    def __init__(self, nifti_path, transposition):
        self.nifti_path = pathlib.Path(nifti_path)
        _raw = (2, 1, 0)
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

    @property
    def shape(self):
        if not hasattr(self, '_shape'):
            img = SimpleITK.ReadImage(self.nifti_path)
            _raw = img.GetSize()
            self._shape = (_raw[self.img_transposition[0]],
                           _raw[self.img_transposition[1]],
                           _raw[self.img_transposition[2]])
        return self._shape

    @property
    def n_raw_dim(self):
        if not hasattr(self, '_n_raw_dim'):
            self._n_raw_dim = len(self.arr.shape)
        return self._n_raw_dim
            
    @property
    def n_channels(self):
        if not hasattr(self, '_n_channels'):
            if self.n_raw_dim == 3:
                self._n_channels = 1
            else:
                self._n_channels = self.arr.shape[-1]
        return self._n_channels
            
    @property
    def arr(self):
        if not hasattr(self, '_arr'):
            self._arr = self._get_arr()
            assert self._arr.shape[:3] == self.shape
        return self._arr

    @property
    def scales(self):
        if not hasattr(self, '_scales'):
            self._scales = self._get_scales()
        return self._scales


    def _get_arr(self) -> np.ndarray:
        """
        Will be cast so that arr.shape matches img.GetSize()
        """
        print("getting array")
        img = SimpleITK.ReadImage(self.nifti_path)
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

    def _get_scales(self) -> tuple:
        """
        Returns dimensions in (1, 2, 3) order as they appear
        in the NIFTI file
        """
        print("reading scales")
        img = SimpleITK.ReadImage(self.nifti_path)
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

        if channel is None:
            channel = 'red'

        if channel not in ('green', 'red', 'blue'):
            raise RuntimeError(
                f"invalid channel: {channel}")

        channel_idx = {'red': 0, 'green': 1, 'blue': 2}[channel]

        if len(self.arr.shape) == 4:
            this_channel = self.arr[:, :, :, channel_idx]
        else:
            this_channel = self.arr

        return {'channel': this_channel,
                'scales': self.scales}


class NiftiArrayCollection(object):

    def __init__(self, nifti_dir_path, transposition=None):
        nifti_dir_path = pathlib.Path(nifti_dir_path)
        if not nifti_dir_path.is_dir():
            raise RuntimeError(
                f"{nifti_dir_path} is not a dir")

        self.transposition = transposition
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
        self.channel_lookup[key] = path

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
        this_path = self.channel_lookup[key]
        nifti_array = NiftiArray(this_path, transposition=self.transposition)

        return nifti_array.get_channel(
                    channel=channel)


def get_nifti_obj(nifti_path, transposition=None):
    nifti_path = pathlib.Path(nifti_path)
    if nifti_path.is_dir():
        return NiftiArrayCollection(nifti_path, transposition=transposition)
    elif nifti_path.is_file():
        return NiftiArray(nifti_path, transposition=transposition)

    raise RuntimeError(
        f"{nifti_path} is neither file nor dir")
