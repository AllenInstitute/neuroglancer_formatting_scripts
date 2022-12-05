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
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()

    if args.table_title is None:
        raise RuntimeError(
            "Must specify table_title")

    if args.output_path is None:
        raise RuntimeError(
            "Must specify output_path")

    data_dir = None
    if args.data_dir is not None:
        data_dir = pathlib.Path(args.data_dir)

    cell_types_bucket = f"{args.s3_location}/cell_types"
    template_bucket = f"{args.s3_location}/avg_template"
    segmentation_bucket = f"{args.s3_location}/ccf_annotations"
    max_count_bucket = f"{args.s3_location}/max_count_image"

    write_celltypes_html(
        output_path=args.output_path,
        cell_types_bucket=cell_types_bucket,
        template_bucket=template_bucket,
        segmentation_bucket=segmentation_bucket,
        max_count_bucket=max_count_bucket,
        cell_types_dir=data_dir/'cell_types',
        title=args.table_title)
    print("wrote html")
    print(args.output_path)

if __name__ == "__main__":
    main()
