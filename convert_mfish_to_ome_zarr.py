import json
import pathlib
import argparse
from neuroglancer_interface.utils.data_utils import (
    write_nii_file_list_to_ome_zarr)


def gene_from_fname(fname):
    params = fname.name.split('_')
    chosen = None
    for p in params:
        try:
            int(p)
        except ValueError:
            chosen = p
            break
    return chosen

def main():
    default_input = '/allen/programs/celltypes/workgroups/rnaseqanalysis/'
    default_input += 'mFISH/michaelkunst/MERSCOPES/mouse/atlas/mouse_1/'
    default_input += 'alignment/mouse1_warpedToCCF_20220906'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--n_processors', type=int, default=4)
    args = parser.parse_args()

    with open(args.config_path, "rb") as in_file:
        config_data = json.load(in_file)
        input_dir = pathlib.Path(config_data["input_dir"])
        output_dir = pathlib.Path(config_data["output_dir"])
        clobber = config_data["clobber"]
        downscale = config_data["downscale"]

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
            raise RuntimeError(msg)
        gene_list.append(gene)

    write_nii_file_list_to_ome_zarr(
        file_path_list=fname_list,
        group_name_list=gene_list,
        output_dir=output_dir,
        downscale=downscale,
        clobber=clobber,
        n_processors=args.n_processors)


if __name__ == "__main__":
    main()
