import argparse
import pathlib

from neuroglancer_interface.modules.celltypes_html import (
    write_celltypes_html)


def main():

    default_anno = '/allen/programs/celltypes/'
    default_anno += 'workgroups/rnaseqanalysis/mFISH'
    default_anno += '/michaelkunst/MERSCOPES/mouse/cluster_anno.csv'

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None,
                        help='data dir for full dataset')
    parser.add_argument('--s3_location', type=str, default=None,
                        help='bucket_name/parent_dir_for_dataset')
    parser.add_argument('--table_title', type=str, default=None)
    args = parser.parse_args()

    if args.table_title is None:
        raise RuntimeError(
            "Must specify table_title")

    data_dir = None
    if args.data_dir is not None:
        data_dir = pathlib.Path(args.data_dir)

    cell_types_bucket = f"{args.s3_location}/cell_types"
    template_bucket = f"{args.s3_location}/avg_template"
    segmenttion_bucket = f"{args.s3_location}/ccf_annotations"

    html_dir = pathlib.Path('html')
    write_celltypes_html(
        output_path=args.output_path,
        cell_types_bucket=cell_types_bucket,
        template_bucket=template_bucket,
        segmentation_bucket=segmentation_bucket,
        cell_types_dir=data_dir/'cell_types',
        title=args.table_title)
    print("wrote html")
    print(args.output_path)

if __name__ == "__main__":
    main()
