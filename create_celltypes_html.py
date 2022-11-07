import argparse
import pathlib

from neuroglancer_interface.modules.celltypes_html import (
    write_celltypes_html)


def main():

    default_anno = '/allen/programs/celltypes/'
    default_anno += 'workgroups/rnaseqanalysis/mFISH'
    default_anno += '/michaelkunst/MERSCOPES/mouse/cluster_anno.csv'

    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', type=str, default=default_anno)
    parser.add_argument('--pass_all', default=False, action='store_true')
    parser.add_argument('--data_dir', type=str, default=None)
    args = parser.parse_args()

    data_dir = None
    if args.data_dir is not None:
        data_dir = pathlib.Path(args.data_dir)

    html_dir = pathlib.Path('html')
    write_celltypes_html(
        output_path=html_dir / 'mouse1_celltype_maps.html',
        annotation_path=pathlib.Path(args.annotation_path),
        pass_all=args.pass_all,
        data_dir=data_dir)
    print("wrote html")

if __name__ == "__main__":
    main()
