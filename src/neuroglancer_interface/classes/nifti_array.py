import numpy as np
import SimpleITK
import pathlib


class NiftiArray(object):
    """
    A class to carry around and self-consistently manipulate
    the image data from a NiftiArray and its geometric metadata
    """

    def __init__(self, nifti_path):
        self.nifti_path = pathlib.Path(nifti_path)
        if not self.nifti_path.is_file():
            raise RuntimError(f"{self.nifti_path} is not a file")

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
            self._read_data()
        return self._arr

    @property
    def scales(self):
        if not hasattr(self, '_scales'):
            self._read_data()
        return self._scales

    def _read_data(self):
        img = SimpleITK.ReadImage(self.nifti_path)
        self._arr = self._get_raw_arr(img)
        self._scales = self._get_raw_scales(img)

    def _get_raw_arr(self, img) -> np.ndarray:
        """
        Will be cast so that arr.shape matches img.GetSize()
        """
        arr = SimpleITK.GetArrayFromImage(img)
        if len(arr.shape) == 3:
            return arr.transpose(2, 1, 0)
        elif len(arr.shape) == 4:
            return arr.transpose(2, 1, 0, 3)
        else:
            raise RuntimeError(
                f"Cannot parse array of shape {arr.shape}")

    def _get_raw_scales(self, img) -> tuple:
        """
        Returns dimensions in (1, 2, 3) order as they appear
        in the NIFTI file
        """
        d1_mm = img.GetMetaData('pixdim[1]')
        d2_mm = img.GetMetaData('pixdim[2]')
        d3_mm = img.GetMetaData('pixdim[3]')
        return (float(d1_mm),
                float(d2_mm),
                float(d3_mm))

    def get_channel(self, channel, transposition=None):

        if channel is None:
            channel = 'red'

        if channel not in ('green', 'red', 'blue'):
            raise RuntimeError(
                f"invalid channel: {channel}")

        channel_idx = {'red': 0, 'green': 1, 'blue': 2}[channel]

        if transposition is not None:
            if len(transposition) != 3:
                raise RuntimeError(
                    "Cannot handle transposition specification "
                    f"of len {len(transposition)}")

        if len(self.arr.shape) == 4:
            this_channel = self.arr[:, :, :, channel_idx]
        else:
            this_channel = self.arr

        if transposition is not None:
            this_channel = this_channel.transpose(transposition)

            output_scales = (self.scales[transposition[0]],
                             self.scales[transposition[1]],
                             self.scales[transposition[2]])
        else:
            output_scales = self.scales

        return {'channel': this_channel,
                'scales': output_scales}


class NiftiArrayCollection(object):

    def __init__(self, nifti_dir_path):
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
        self.channel_lookup[key] = path

    def get_channel(self, channel, transposition=None):
        if channel is None:
            channel = 'red'
        this_path = self.channel_lookup[key]
        nifti_array = NiftiArray(this_path)

        return nifti_array.get_channel(
                    channel=channel,
                    transposition=transposition)


def get_nifti_obj(nifti_path):
    nifti_path = pathlib.Path(nifti_path)
    if nifti_path.is_dir():
        return NiftiArrayCollection(nifti_path)
    elif nifti_path.is_file():
        return NiftiArray(nifti_path)

    raise RuntimeError(
        f"{nifti_path} is neither file nor dir")
