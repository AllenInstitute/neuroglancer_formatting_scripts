import multiprocessing
import numpy as np
import pathlib
import SimpleITK


def get_mask_lookup(
        nifti_config_list,
        n_processors):
    """
    Parameters
    -----------
    nifti_config_list:
        List of configs like
            {'path': path/to/nifti/file
             'structure': name_of_structure}

    n_processors:
        Number of independent workers to spin up

    Returns
    -------
    Dict
        maps name_to_structure to a dict for each mask containing
        'path'
        'shape'
        'pixels'
        (see get_mask_from_NIFTI for documentation)
    """
    # make sure all structure tags are unique
    structure_tag_names = set()
    for config in nifti_config_list:
        if config['structure'] in structure_tag_names:
            raise RuntimeError(
                f"structure name {config['structure']} occurs "
                f"more than once")
        structure_tag_names.add(config['structure'])

    mgr = multiprocessing.Manager()
    output_dict = mgr.dict()
    output_lock = mgr.Lock()
    process_list = []

    n_per = max(1, len(nifti_config_list)//n_processors)
    for i_batch in range(n_processors):
        i0 = i_batch*n_per
        if i_batch == n_processors-1:
            i1 = len(nifti_config_list)
        else:
            i1 = i0 + n_per
        batch = nifti_config_list[i0:i1]
        p = multiprocessing.Process(
                target=_get_mask_lookup_worker,
                kwargs={
                    'nifti_config_list': batch,
                    'output_dict': output_dict,
                    'output_lock': output_lock})
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()
    return dict(output_dict)


def _get_mask_lookup_worker(
        nifti_config_list,
        output_dict,
        output_lock):

    these = dict()
    for config in nifti_config_list:
        mask = get_mask_from_NIFTI(
            nifti_path=config['path'])
        these[config['structure']] = mask

    with output_lock:
        output_dict.update(these)


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
