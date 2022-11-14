import pathlib
import argparse
from neuroglancer_interface.utils.data_utils import (
    write_nii_file_list_to_ome_zarr)


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

    assert input_dir.is_dir()

    fname_list = [n for n in input_dir.rglob('*nii.gz')]

    fname_list.sort()

    genes_loaded = set()
    gene_list = []
    for fname in fname_list:
        gene = gene_from_fname(fname)
        if gene in genes_loaded:
            msg = f"{gene} appears twice\n"
            msg += f"{fname}\n"
            msg += f"{gene_to_path[gene]}"
            raise RuntimeError(msg)
        gene_list.append(gene)

    write_nii_file_list_to_ome_zarr(
        file_path_list=fname_list,
        group_name_list=gene_list,
        output_dir=output_dir,
        downscale=args.downscale,
        clobber=args.clobber)


if __name__ == "__main__":
    main()
