from cloudvolume import CloudVolume
import argparse
import shutil
import PIL
import pathlib
import numpy as np
import json
import skimage.transform


def make_info_file(
        resolution_xyz,
        volume_size_xyz,
        layer_dir,
        downscale_list = (1, 2, 4, 8)):
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


    base_resolution = [10000, 10000, 100000]

    info = dict()
    info["data_type"] = "uint8"
    info["num_channels"] = 3
    info["type"] = "image"

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


def read_image_to_cloud(image_path_list,
                        layer_dir,
                        key,
                        chunk_size,
                        base_shape,
                        downscale_shape):

    np_base_shape = (base_shape[1],
                     base_shape[0],
                     3)

    np_scaled_shape = (downscale_shape[1],
                       downscale_shape[0],
                       3)

    this_dir = layer_dir / key
    if not this_dir.exists():
        this_dir.mkdir()
    assert this_dir.is_dir()

    dx = chunk_size[0]
    dy = chunk_size[1]

    if not isinstance(image_path_list, list):
        image_path_list = [image_path_list]

    for zz, image_path in enumerate(image_path_list):
        data = read_and_pad_image(
                    image_path=image_path,
                    np_target_shape=np_base_shape)

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
                this_file = this_dir / f"{x0}-{x1}_{y0}-{y1}_{zz}-{zz+1}"
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


def process_image(image_path_list, image_dir):

    if not isinstance(image_path_list, list):
        image_path_list = [image_path_list]

    volume_shape = get_volume_shape(image_path_list)

    info_data = make_info_file(
        resolution_xyz=(10000, 10000, 100000),
        volume_size_xyz=volume_shape,
        layer_dir=image_dir)

    for scale in info_data["scales"]:
        img_cloud = read_image_to_cloud(
            image_path_list=image_path_list,
            layer_dir=image_dir,
            key=scale["key"],
            chunk_size=scale["chunk_sizes"][0],
            base_shape=volume_shape,
            downscale_shape=scale["size"])

def main():

    parser = argparse.ArgumentParser()
    #parser.add_argument('--ish_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--clobber', default=False, action='store_true')
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    if output_dir.exists():
        if not args.clobber:
            raise RuntimeError(f"{output_dir} exists")
        else:
            shutil.rmtree(output_dir)
    output_dir.mkdir()

    image_dir = pathlib.Path('/Users/scott.daniel/KnowledgeBase/ish_example/data')
    image_path_list = [n for n in image_dir.rglob('10004*')]
    image_path_list.sort()
    for p in image_path_list:
        print(p)

    process_image(image_path_list=image_path_list,
                  image_dir=output_dir)


if __name__ == "__main__":
    main()
    #sfd need to get colors right
    #   Note: the shader code for the Seung lab branch was not the same
    #   as the shader code for google; look at some examples using
    #   sfd-eastern-bucket (...)
