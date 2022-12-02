import argparse
from neuroglancer_interface.modules.ccf_annotation_formatting import (
    format_ccf_annotations)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', type=str, default=None,
                        help='path to text file with label names in it')
    parser.add_argument('--segmentation_path', type=str, default=None,
                        help='path to the segmentation .nii.gz file')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--clobber', default=False, action='store_true')
    args = parser.parse_args()

    format_ccf_annotations(
        annotation_path=args.annotation_path,
        segmentation_path=args.segmentation_path,
        clobber=args.clobber,
        output_dir=args.output_dir)

if __name__ == "__main__":
    main()
# just need to unzip files
# figure out how to host atlas from s3
# figure out if generating the 3D mesh is really very hard
