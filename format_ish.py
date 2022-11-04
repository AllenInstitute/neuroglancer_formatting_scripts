from cloudvolume import CloudVolume
import argparse
import shutil
import PIL
import pathlib
import numpy as np
import json
import multiprocessing
import skimage.transform

try:
    import affpyramid
    has_aff = True
except ModuleNotFoundError:
    has_aff = False


def make_info_file(
        resolution_xyz,
        volume_size_xyz,
        layer_dir,
        image_config_list,
        downscale_list = (1, 2, 4, 8),
        metadata=None):
    """
    Shamelessly copied from

    https://github.com/PrincetonUniversity/lightsheet_helper_scripts/blob/master/neuroglancer/brodylab_MRI_atlas_customizations.ipynb

    Make an JSON-formatted file called the "info" file
    for use with the precomputed data format. 
    Precomputed is one of the formats that Neuroglancer can read in.  
    --- parameters ---
    resolution_xyz:      A tuple representing the size of the pixels (dx,dy,dz) 
                         in nanometers, e.g. (20000,20000,5000) for 20 micron x 20 micron x 5 micron
    
    volume_size_xyz:     A tuple representing the number of pixels in each dimension (Nx,Ny,Nz)

                         
    layer_dir:           The directory where the precomputed data will be
                         saved
    """


    actual_downscale = []
    for v in downscale_list:
        if volume_size_xyz[0] % v == 0 and volume_size_xyz[1] % v == 0:
            actual_downscale.append(v)


    base_resolution = resolution_xyz

    info = dict()
    info["data_type"] = "uint8"
    info["num_channels"] = 3
    info["type"] = "image"

    z_to_image_lookup = dict()
    for ii in range(len(image_config_list)):
        config = image_config_list[ii]
        z_to_image_lookup[ii] = config

    info["z_to_LIMS_metadata"] = z_to_image_lookup
    info["metadata"] = metadata

    scale_list = []
    for downscale in actual_downscale:
        this_resolution = [base_resolution[0]*downscale,
                           base_resolution[1]*downscale,
                           base_resolution[2]]
        this_size = [volume_size_xyz[0]//downscale,
                     volume_size_xyz[1]//downscale,
                     volume_size_xyz[2]]
        this_scale = dict()
        this_scale["key"] = f"{this_resolution[0]}_"
        this_scale["key"] += f"{this_resolution[1]}_"
        this_scale["key"] += f"{this_resolution[2]}"
        this_scale["encoding"] = "raw"
        this_scale["resolution"] = this_resolution
        this_scale["size"] = this_size
        this_scale["chunk_sizes"] = [[512, 512, 1]]
        scale_list.append(this_scale)

    info["scales"] = scale_list
    with open(f"{layer_dir}/info", "w") as out_file:
        out_file.write(json.dumps(info, indent=2))

    return info

def read_and_pad_image(
        image_path,
        np_target_shape):
    """
    np_target_shape is the shape of the np.array
    we want to return; will be the transpose of the
    img.size
    """

    result = np.zeros(np_target_shape, dtype=np.uint8)
    with PIL.Image.open(image_path, 'r') as img:
        result[:img.size[1],
               :img.size[0],
               :] = np.array(img)

    return result


def read_and_pad_image_config(
        image_config,
        image_path,
        np_target_shape):
    """
    np_target_shape is the shape of the np.array
    we want to return; will be the transpose of the
    img.size
    """
    r0 = image_config['y']
    c0 = image_config['x']
    r1 = r0 + image_config['height']
    c1 = c0 + image_config['width']

    result = np.zeros(np_target_shape, dtype=np.uint8)
    aff = affpyramid.AffPyramid(image_path)
    aff_data = aff.get_tier(aff.num_tiers-1)
    aff_data = aff_data[r0:r1, c0:c1]
    assert aff_data.dtype == np.uint8

    result[:aff_data.shape[0],
           :aff_data.shape[1],
           :] = aff_data

    return result


def write_image_to_cloud(
        layer_dir,
        key,
        chunk_size,
        downscale_shape,
        data,
        zz_idx):


    np_scaled_shape = (downscale_shape[1],
                       downscale_shape[0],
                       3)

    this_dir = layer_dir / key
    if not this_dir.exists():
        this_dir.mkdir()
    assert this_dir.is_dir()

    dx = chunk_size[0]
    dy = chunk_size[1]

    if not np.allclose(data.shape, np_scaled_shape):
        print(f"resizing {data.shape} -> {np_scaled_shape}")
        data = skimage.transform.resize(
                    data,
                    np_scaled_shape,
                    preserve_range=True,
                    anti_aliasing=True)

        data = np.round(data).astype(np.uint8)

    for x0 in range(0, data.shape[1], dx):
        x1 = min(data.shape[1], x0+dx)
        for y0 in range(0, data.shape[0], dy):
            y1 = min(data.shape[0], y0+dy)
            this_file = this_dir / f"{x0}-{x1}_{y0}-{y1}_{zz_idx}-{zz_idx+1}"
            with open(this_file, "wb") as out_file:
                this_data = data[y0:y1, x0:x1, :].transpose(1, 0, 2).tobytes("F")
                out_file.write(this_data)


def get_volume_shape(image_path_list):
    dx_vals = []
    dy_vals = []
    for image_path in image_path_list:
        with PIL.Image.open(image_path, 'r') as img:
            dx_vals.append(img.size[0])
            dy_vals.append(img.size[1])

    return (max(dx_vals), max(dy_vals), len(image_path_list))


def get_volume_shape_from_config(image_config_list):
    width = max([el['width'] for el in image_config_list])
    height = max([el['height'] for el in image_config_list])
    return (width, height, len(image_config_list))


def _process_image(
        image_config_lookup,
        image_dir,
        volume_shape,
        info_data):
    """
    image_config_lookup maps zz_idx to image config
    """

    if len(image_config_lookup) == 0:
        return

    np_base_shape = (volume_shape[1],
                     volume_shape[0],
                     3)


    raw_data_path = None
    raw_data = None

    for zz_idx in image_config_lookup:

        image_config = image_config_lookup[zz_idx]

        image_path = pathlib.Path(image_config['storage_directory'])
        image_path = image_path / image_config['zoom']
        image_path = str(image_path.resolve().absolute())

        if raw_data_path is None or image_path != raw_data_path:
            raw_data = read_and_pad_image_config(
                        image_config=image_config,
                        image_path=image_path,
                        np_target_shape=np_base_shape)
            raw_data_path = image_path

        for scale in info_data["scales"]:
            img_cloud = write_image_to_cloud(
                layer_dir=image_dir,
                key=scale["key"],
                chunk_size=scale["chunk_sizes"][0],
                downscale_shape=scale["size"],
                data=raw_data,
                zz_idx=zz_idx)


def process_image(
        image_config_list,
        image_dir,
        n_processors,
        metadata=None):


    if not isinstance(image_config_list, list):
        image_config_list = [image_config_list]

    #volume_shape = get_volume_shape(image_path_list)

    baseline_resolution = None
    for image_config in image_config_list:
        if baseline_resolution is None:
            baseline_resolution = image_config['resolution']
        else:
            if not np.allclose(baseline_resolution,
                               image_config['resolution']):
                raise RuntimeError(
                    "\nInconsistent resolutions\n"
                    f"{baseline_resolution} != {image_config['resolution']}")

    volume_shape = get_volume_shape_from_config(image_config_list)

    baseline_xyz = int(np.round(baseline_resolution*1000).astype(int))
    resolution_xyz = (baseline_xyz, baseline_xyz, baseline_xyz)

    info_data = make_info_file(
        resolution_xyz=resolution_xyz,
        volume_size_xyz=volume_shape,
        layer_dir=image_dir,
        image_config_list=image_config_list,
        metadata=metadata)

    process_list = []
    sub_lists = []
    for ii in range(n_processors):
        sub_lists.append(dict())
    for ii in range(len(image_config_list)):
        jj = ii % n_processors
        sub_lists[jj][ii] = image_config_list[ii]

    for ii in range(n_processors):
        p = multiprocessing.Process(
                target=_process_image,
                kwargs={
                    'image_config_lookup': sub_lists[ii],
                    'image_dir': image_dir,
                    'volume_shape': volume_shape,
                    'info_data': info_data})
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()


def process_all_images(
        image_config_list,
        image_dir,
        n_processors,
        clobber=False,
        metadata=None):
    """
    Group configs by image_series_id; create sub-directories
    for each of those. Load images into those directories.
    """

    # find all  of the unique image_series_id values
    valid_ids = set()
    for image_config in image_config_list:
        valid_ids.add(image_config["image_series_id"])


    # create a destination directory for each images_series_id
    for image_series_id in valid_ids:
        this_dir = image_dir / f"{image_series_id}"
        if this_dir.exists():
            if not clobber:
                raise RuntimeError(f"{this_dir.resolve().absolute()} exists")
            else:
                shutil.rmtree(this_dir)
        this_dir.mkdir()

    # create a dict mapping image_series_id to a list of
    # image configs sorted by image_series_id
    config_lookup = dict()
    for image_series_id in valid_ids:
        this_idx = []
        this_config = []
        for image_config in image_config_list:
            if not image_config["image_series_id"] == image_series_id:
                continue
            this_config.append(image_config)
            this_idx.append(image_config["specimen_tissue_index"])
        this_idx = np.array(this_idx)
        sorted_dex = np.argsort(this_idx)
        final_list = []
        for ii in sorted_dex:
            final_list.append(this_config[ii])
        config_lookup[image_series_id] = final_list

    # actually process each image_series_id
    id_list = list(config_lookup.keys())
    id_list.sort()
    for image_series_id in id_list:
        if metadata is None:
            this_metadata = None
        else:
            this_metadata = metadata[str(image_series_id)]


        this_dir = image_dir / f"{image_series_id}"
        print(f"loading {this_dir.resolve().absolute()}")
        process_image(
            image_config_list=config_lookup[image_series_id],
            image_dir=this_dir,
            n_processors=n_processors,
            metadata=metadata)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--clobber', default=False, action='store_true')
    parser.add_argument('--n_processors', type=int, default=4)
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    if output_dir.exists():
        if not args.clobber:
            raise RuntimeError(f"{output_dir} exists")
        else:
            shutil.rmtree(output_dir)
    output_dir.mkdir()

    with open(args.config_path, 'rb') as in_file:
        global_config = json.load(in_file)

    image_config_list = global_config['configs']
    metadata = global_config['metadata']

    process_all_images(
            image_config_list=image_config_list,
            image_dir=output_dir,
            n_processors=args.n_processors,
            clobber=args.clobber,
            metadata=metadata)


if __name__ == "__main__":
    main()
    #sfd need to get colors right
    #   Note: the shader code for the Seung lab branch was not the same
    #   as the shader code for google; look at some examples using
    #   sfd-eastern-bucket (...)
