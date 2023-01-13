import glymur
import h5py
import tempfile
import pathlib
import time


def get_jp2_config(
        config_list):
    """
    config_list is a list of dicts containign the config information
    for the jp2 files being stacked.

    This method checks the consistency of shape and dtype across
    those files and returns them in a dict.
    """

    shape = None
    dtype = None
    for config in config_list:
        jp2 = glymur.Jp2k(config['image_path'])
        if shape is None:
            shape = jp2.shape
            dtype = jp2.dtype
        else:
            if jp2.shape != shape:
                raise RuntimeError("shape mismatch")
            if jp2.dtype != dtype:
                raise RuntimeError("dtype mismatch")
    return {"shape": shape, "dtype": dtype}


def write_data_to_hdf5(
        config_list,
        tmp_dir)
    """
    config_list is a list of dicts containign the config information
    for the jp2 files being stacked.

    Return the path to the HDF5 file
    """

    h5_path = pathlib.Path(
                tempfile.mkstemp(
                    dir=tmp_dir,
                    prefix='ome_zarr_temp_'
                    suffix='.h5')[1])

    try:
        _write_data_to_hdf5(
            config_list=config_list,
            h5_path=h5_path,
            clobber=False)
    except:
        if h5_path.exists():
            h5_path.unlink()
        raise

    return h5_path


def _write_data_to_hdf5(
        config_list,
        h5_path,
        clobber=False):
    """
    config_list is a list of dicts containign the config information
    for the jp2 files being stacked.
    """

    if h5_path.exists():
        if not clobber:
            raise RuntimeError(f"{h5_path} exists already")

    jp2_config = get_jp2_config(config_list)

    nz = len(config_list)
    shape = (jp2_config.shape[0],
             jp2_config.shape[1],
             nz)

    chunks = (max(1, min(1000, shape[0]//4)),
              max(1, min(1000, shape[0]//4)),
              1)

    ct = 0
    t0 = time.time()
    print(f"writing data to {h5_path.resolve().absolute()}")
    with h5py.File(h5_path, "w") as out_file:
        for data_name in ("red", "green"):
            out_file.create_dataset(
                data_name,
                shape=shape,
                dtype=jp2_config['dtype'],
                chunks=chunks,
                compression='gzip')

        for iz in range(config_list):
            raw_data = glymur.Jp2k(config_list[iz]['image_path'])[:, :, :]
            out_file['red'][:, :, iz] = raw_data[:, :, 0]
            out_file['green'][:, :, iz] = raw_data[:, :, 1]

            duration = (time.time()-t0)/3600.0
            per = duration/(iz+1)
            pred = per*nz
            remain = pred-duration
            print(f"{ct} written in {duration:.2e} hrs; "
                  f"predict {remain:.2e} of {pred:.2e} remaining")
