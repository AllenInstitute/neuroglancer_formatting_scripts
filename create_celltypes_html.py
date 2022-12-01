import argparse
import pathlib

from neuroglancer_interface.modules.celltypes_html import (
    write_celltypes_html)


def main():

    default_anno = '/allen/programs/celltypes/'
    default_anno += 'workgroups/rnaseqanalysis/mFISH'
    default_anno += '/michaelkunst/MERSCOPES/mouse/cluster_anno.csv'

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--cell_types_bucket', type=str, default=None)
    parser.add_argument('--template_bucket', type=str, default=None)
    parser.add_argument('--segmentation_bucket', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--table_title', type=str, default=None)
    parser.add_argument('--x_mm', type=float, default=0.01)
    parser.add_argument('--y_mm', type=float, default=0.01)
    parser.add_argument('--z_mm', type=float, default=0.1)
    parser.add_argument('--n_processors', type=int, default=6)
 
    args = parser.parse_args()

    data_dir = None
    if args.data_dir is not None:
        data_dir = pathlib.Path(args.data_dir)

    html_dir = pathlib.Path('html')
    write_celltypes_html(
        output_path=args.output_path,
        cell_types_bucket=args.cell_types_bucket,
        template_bucket=args.template_bucket,
        segmentation_bucket=args.segmentation_bucket,
        data_dir=data_dir,
        title=args.table_title,
        x_mm=args.x_mm,
        y_mm=args.y_mm,
        z_mm=args.z_mm,
        n_processors=args.n_processors)
    print("wrote html")
    print(args.output_path)

if __name__ == "__main__":
    main()
