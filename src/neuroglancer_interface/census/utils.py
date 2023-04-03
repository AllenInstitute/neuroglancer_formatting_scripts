import numpy as np
import pathlib
import SimpleITK


def get_mask_from_NIFTI(
        nifti_path):
    """
    Parameters
    ----------
    nifti_path:
         Path to the NIFTI structure mask file
    
    Returns
    -------
    Dict:
        'shape' -- the array shape of the mask
        'pixels' -- result of np.where(arr==1)
        'path' -- path to the file
    """
    nifti_path = pathlib.Path(nifti_path)

    arr = SimpleITK.GetArrayFromImage(
            SimpleITK.ReadImage(nifti_path))
    
    result = dict()
    result['path'] = str(nifti_path.resolve().absolute())
    result['shape'] = arr.shape
    result['pixels'] = np.where(arr==1)
    return result
