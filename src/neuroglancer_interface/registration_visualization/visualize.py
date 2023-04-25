import argparse
import json
import pathlib
import shutil
import SimpleITK
import tempfile

from neuroglancer_interface.utils.utils import (
    _clean_up)

from neuroglancer_interface.utils.data_utils import (
    create_root_group)

from neuroglancer_interface.modules.ccf_multiscale_annotations import (
    write_out_ccf)

from neuroglancer_interface.utils.data_utils import (
    write_nii_file_list_to_ome_zarr,
    write_nii_to_group)


def print_status(msg):
    print(f"===={msg}====")


def process_registration(
        config_data,
        tmp_dir):
    """
    Parameters
    ----------
    config_data:
        A dict containing the data that is to be processed
    tmp_dir:
        pathlib.Path pointing to temporary directory where processed
        data will be written out.
    """

    junk_dir = tempfile.mkdtemp(dir=tmp_dir)

    ccf_dir = None
    template_dir = None

    ccf_config = config_data["ccf"]
    if len(ccf_config) > 0:
        ccf_dir = tmp_dir /"ccf"

        if not ccf_dir.exists():
            ccf_dir.mkdir(parents=True)

        for ccf_data in ccf_config:
            print_status(f"processing {ccf_data['nii_path']}")
            this_dir = ccf_dir / ccf_data["tag"]
            if not this_dir.exists():
                this_dir.mkdir(parents=True)
            write_out_ccf(
                segmentation_path_list = [
                    pathlib.Path(ccf_data["nii_path"])],
                label_path=ccf_data["label_path"],
                output_dir=this_dir,
                use_compression=True,
                compression_blocksize=23,
                chunk_size=(64, 64, 64),
                do_transposition=False,
                tmp_dir=junk_dir,
                downsampling_cutoff=64)
                    
    template_config = config_data["template"]
    if len(template_config) > 0:
        template_dir = tmp_dir / "template"
        template_group = create_root_group(output_dir=template_dir)

        for template_data in template_config:
            print_status(f"processing {template_data['nii_path']}")
            write_nii_to_group(
                root_group=template_group,
                group_name=template_data["tag"],
                nii_file_path=template_data["nii_path"],
                downscale_cutoff=64,
                default_chunk=128,
                channel='red',
                do_transposition=False)     

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--tmp_dir', type=str, default=None)
    parser.add_argument('--clean_up', action='store_true', default=False)
    args = parser.parse_args()

    tmp_dir = pathlib.Path(
        tempfile.mkdtemp(dir=args.tmp_dir))
    print_status(f"writing data to {tmp_dir}")

    config_data = json.load(open(args.config_path, 'rb'))
    process_registration(
        config_data=config_data,
        tmp_dir=tmp_dir)

    if args.clean_up:
        pring_status(f"cleaning up {tmp_dir}")
        _clean_up(tmp_dir)


if __name__ == "__main__":
    main()
