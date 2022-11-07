import argparse
import pathlib

from neuroglancer_interface.modules.mfish_html import (
    write_mfish_html)


def main():
    html_dir = pathlib.Path('html')
    write_mfish_html(output_path=html_dir / 'mouse1_mfish_maps.html')


if __name__ == "__main__":
    main()
