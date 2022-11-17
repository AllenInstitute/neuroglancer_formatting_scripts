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
    parser.add_argument('--output_path', type=str, default=default_output)
    parser.add_argument('--gene_list_path', type=str, default=default_genes)
    parser.add_argument('--mfish_bucket', type=str,
                        default='mouse1-mfish-prototype')
    parser.add_argument('--segmentation_bucket', type=str,
                        default='mouse1-atlas-prototype')
    parser.add_argument('--template_bucket', type=str,
                        default='mouse1-template-prototype/template')
    parser.add_argument('--page_title', type=str,
                        default="Mouse1 MFISH transcript count maps")
    args = parser.parse_args()

    write_mfish_html(
        output_path=pathlib.Path(args.output_path),
        gene_list_path=pathlib.Path(args.gene_list_path),
        mfish_bucket=args.mfish_bucket,
        segmentation_bucket=args.segmentation_bucket,
        template_bucket=args.template_bucket,
        html_title=args.page_title)


if __name__ == "__main__":
    main()
