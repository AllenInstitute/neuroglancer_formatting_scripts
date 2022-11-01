from cloudvolume import CloudVolume
import argparse
import shutil
import PIL
import pathlib
import numpy as np
import json


def make_info_file(
        resolution_xyz,
        volume_size_xyz,
        layer_dir):
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

    info = dict()
    info["data_type"] = "uint8"
    info["num_channels"] = 3
    info["type"] = "image"

    main_scale = dict()
    main_scale["key"] = "10000_10000_100000"
    main_scale["encoding"] = "raw"
    main_scale["resolution"] = [10000, 10000, 100000]
    main_scale["size"] = volume_size_xyz
    main_scale["chunk_sizes"] = [[512, 512, 1]]

    info["scales"] = [main_scale]
    with open(f"{layer_dir}/info", "w") as out_file:
        out_file.write(json.dumps(info, indent=2))


def read_image_to_cloud(image_path,
                        layer_dir,
                        key,
                        chunk_size):
    this_dir = layer_dir / key
    if not this_dir.exists():
        this_dir.mkdir()
    assert this_dir.is_dir()

    dx = chunk_size[0]
    dy = chunk_size[1]

    with PIL.Image.open(image_path, 'r') as img:
        data = np.array(img)
        for x0 in range(0, data.shape[1], dx):
            x1 = min(data.shape[1], x0+dx)
            for y0 in range(0, data.shape[0], dy):
                y1 = min(data.shape[0], y0+dy)
                this_file = this_dir / f"{x0}-{x1}_{y0}-{y1}_{0}-1"
                with open(this_file, "wb") as out_file:
                    this_data = data[y0:y1, x0:x1, :].transpose(1, 0, 2).tobytes("F")
                    out_file.write(this_data)


def process_image(image_path, image_dir):
    with PIL.Image.open(image_path, 'r') as img:
        img_size = img.size
    img_shape = [img.size[0], img.size[1], 1]
    img_cloud = make_info_file(
        resolution_xyz=(10000, 10000, 100000),
        volume_size_xyz=img_shape,
        layer_dir=image_dir)

    img_cloud = read_image_to_cloud(
        image_path=image_path,
        layer_dir=image_dir,
        key="10000_10000_100000",
        chunk_size=[512, 512, 1])

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--ish_path', type=str, default=None)
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

    process_image(image_path=args.ish_path,
                  image_dir=output_dir)


if __name__ == "__main__":
    main()
    #sfd need to get colors right
    #   Note: the shader code for the Seung lab branch was not the same
    #   as the shader code for google; look at some examples using
    #   sfd-eastern-bucket (...)
