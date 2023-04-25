import argparse
import json
import pathlib
import shutil
import SimpleITK
import tempfile
import datetime

from neuroglancer_interface.utils.utils import (
    _clean_up)

from neuroglancer_interface.utils.data_utils import (
    create_root_group)

from neuroglancer_interface.modules.ccf_multiscale_annotations import (
    write_out_ccf)

from neuroglancer_interface.utils.data_utils import (
    write_nii_file_list_to_ome_zarr,
    write_nii_to_group)

from neuroglancer_interface.utils.s3_utils import (
    upload_to_bucket)

from neuroglancer_interface.utils.url_utils import(
    get_final_url,
    get_segmentation_layer,
    get_template_layer)

from neuroglancer_interface.utils.html_utils import (
    write_basic_table)


def print_status(msg):
    print(f"===={msg}====")


def convert_registration_data(
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

    datasets_created = dict()
    datasets_created['ccf'] = []
    datasets_created['template'] = []

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
            datasets_created['ccf'].append(
                {'tag': ccf_data['tag'],
                 's3': str(this_dir.relative_to(tmp_dir)),
                 'path': this_dir})

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

            datasets_created['template'].append(
                {'tag': template_data['tag'],
                 's3': f"template/{template_data['tag']}",
                 'path': template_dir/template_data['tag']})

    return datasets_created

def upload_data(
        processed_datasets,
        bucket_prefix='scratch/230425/junk2',
        bucket_name='neuroglancer-vis-prototype'):

    data_to_load = []
    for k in ('ccf', 'template'):
        for el in processed_datasets[k]:
            data_to_load.append(el)
    upload_to_bucket(
        data_list=data_to_load,
        bucket_name=bucket_name,
        bucket_prefix=bucket_prefix,
        n_processors=6)


def create_url(
        processed_datasets,
        bucket_prefix,
        bucket_name):

    template_layer_list = []
    for ii, el in enumerate(processed_datasets['template']):
        if 'merfish' in el['tag'].lower():
            range_max = 150
        else:
            range_max = 700

        template = get_template_layer(
            template_bucket=f"{bucket_name}/{bucket_prefix}/{el['s3']}",
            public_name=el['tag'],
            range_max=range_max)

        if ii == 0:
            template['visible'] = True
        else:
            template['visible'] = False
        template_layer_list.append(template)

    ccf_layer_list = []
    for ii, el in enumerate(processed_datasets['ccf']):
        ccf = get_segmentation_layer(
            segmentation_bucket=f"{bucket_name}/{bucket_prefix}/{el['s3']}",
            segmentation_name=el['tag'])
        if ii == 0:
            ccf['visible'] = True
        else:
            ccf['visible'] = False
        ccf_layer_list.append(ccf)

    url = get_final_url(
        image_layer_list = [],
        template_layer=template_layer_list,
        segmentation_layer=ccf_layer_list)

    return url


def run(
    config_path,
    tmp_dir,
    bucket_prefix='junkURL',
    output_path='registration_test.html'):

    bucket_name = 'neuroglancer-vis-prototype'
    now = datetime.datetime.now()
    suffix = f"{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}"
    bucket_prefix = f"registration-vis/{bucket_prefix}/{suffix}"

    config_data = json.load(open(config_path, 'rb'))
    data_created = convert_registration_data(
        config_data=config_data,
        tmp_dir=tmp_dir)
    print(data_created)

    upload_data(processed_datasets=data_created,
        bucket_name=bucket_name,
        bucket_prefix=bucket_prefix)

    url = create_url(
        processed_datasets=data_created,
        bucket_name=bucket_name,
        bucket_prefix=bucket_prefix)

    write_basic_table(
        output_path=output_path,
        title='Registration visualization',
        key_to_link={'view': url},
        key_order=['view'],
        key_to_other_cols={'view': {'names': [], 'values': []}},
        div_name='view')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=None,
        help='path to the JSONized config')
    parser.add_argument('--bucket_prefix', type=str, default=None,
        help='human-readable name for director in bucket where data will be '
        'loaded')
    parser.add_argument('--output_path', type=str, default=None,
        help='path to html file that will be written')
    parser.add_argument('--tmp_dir', type=str, default=None,
        help='path to a directory where temporary data products will be written')
    parser.add_argument('--clean_up', action='store_true', default=False,
        help='if run with this flag, temp dir will be automatically cleaned up '
        '(even if an Exception occurs)')
    args = parser.parse_args()

    tmp_dir = pathlib.Path(
        tempfile.mkdtemp(dir=args.tmp_dir))
    print_status(f"writing data to {tmp_dir}")

    try:
        run(
            config_path=args.config_path,
            bucket_prefix=args.bucket_prefix,
            output_path=args.output_path,
            tmp_dir=tmp_dir)

    finally:
        if args.clean_up:
            print_status(f"cleaning up {tmp_dir}")
            _clean_up(tmp_dir)


if __name__ == "__main__":
    main()
