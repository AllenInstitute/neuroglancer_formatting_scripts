import argparse
import pathlib

from neuroglancer_interface.modules.mfish_html import (
    write_mfish_html)


def main():

    html_dir = pathlib.Path('html')
    default_output = html_dir / 'mouse1_mfish_maps.html'
    default_output = str(default_output.resolve().absolute())

    default_genes = pathlib.Path('data/mouse1_gene_list.json')
    default_genes = str(default_genes.resolve().absolute())

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None,
                        help='data dir for full dataset')
    parser.add_argument('--s3_location', type=str, default=None,
                        help='bucket_name/parent_dir_for_dataset')
    parser.add_argument('--table_title', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()

    mfish_bucket = f"{args.s3_location}/mfish_heatmaps"
    segmentation_bucket = f"{args.s3_location}/ccf_annotations"
    template_bucket = f"{args.s3_location}/avg_template"

    write_mfish_html(
        output_path=pathlib.Path(args.output_path),
        mfish_bucket=args.mfish_bucket,
        segmentation_bucket=segmentation_bucket,
        template_bucket=template_bucket,
        html_title=args.table_title,d
        data_dir=pathilb.Path(args.data_dir))


if __name__ == "__main__":
    main()
