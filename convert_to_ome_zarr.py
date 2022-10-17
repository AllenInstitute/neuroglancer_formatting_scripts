import pathlib
from ome_zarr.io import parse_url
import zarr
import numpy as np
import SimpleITK
import argparse
import shutil
from utils import write_array_to_group


def gene_from_fname(fname):
    gene = fname.name.split('_')[1]
    return gene

def main():
    default_input = '/allen/programs/celltypes/workgroups/rnaseqanalysis/'
    default_input += 'mFISH/michaelkunst/MERSCOPES/mouse/atlas/mouse_1/'
    default_input += 'alignment/mouse1_warpedToCCF_20220906'

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--input_dir', type=str, default=default_input)
    parser.add_argument('--clobber', default=False, action='store_true')
    parser.add_argument('--downscale', type=int, default=2)
    args = parser.parse_args()

    assert args.output_dir is not None
    assert args.input_dir is not None

    output_dir = pathlib.Path(args.output_dir)
    input_dir = pathlib.Path(args.input_dir)
    if output_dir.exists():
        if not args.clobber:
            raise RuntimeError(f"{output_dir} exists")
        else:
            shutil.rmtree(output_dir)
    assert not output_dir.exists()
    output_dir.mkdir()
    assert output_dir.is_dir()

    assert input_dir.is_dir()

    fname_list = [n for n in input_dir.rglob('*nii.gz')]

    store = parse_url(output_dir, mode="w").store
    root_group = zarr.group(store=store)

    fname_list.sort()

    gene_to_path = dict()
    for fname in fname_list:
        gene = gene_from_fname(fname)
        if gene in gene_to_path:
            msg = f"{gene} appears twice\n"
            msg += f"{fname}\n"
            msg += f"{gene_to_path[gene]}"
            raise RuntimeError(msg)
        gene_to_path[gene] = fname

    gene_list = list(gene_to_path.keys())
    gene_list.sort()

    for gene in gene_list:
        fname = gene_to_path[gene]
        print(f"processing {fname.name}")
        this_group = root_group.create_group(f"{gene}")
        img = SimpleITK.ReadImage(fname)

        arr = SimpleITK.GetArrayFromImage(img)
        arr = arr.transpose(2, 1, 0)

        x_scale = float(img.GetMetaData('pixdim[1]'))
        y_scale = float(img.GetMetaData('pixdim[2]'))
        z_scale = float(img.GetMetaData('pixdim[3]'))

        write_array_to_group(
            arr=arr,
            group=this_group,
            x_scale=x_scale,
            y_scale=y_scale,
            z_scale=z_scale,
            downscale=args.downscale)


if __name__ == "__main__":
    main()
