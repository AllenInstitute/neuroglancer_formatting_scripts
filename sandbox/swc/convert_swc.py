import pathlib
import json
from logans_code import binarize_swc


def convert_swc(
        swc_path,
        output_dir):
    swc_path = pathlib.Path(swc_path)
    if not swc_path.is_file():
        raise RuntimeError(f"{swc_file} is not file")

    output_dir = pathlib.Path(output_dir)
    properties_dir = output_dir / 'segment_properties'
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    assert output_dir.is_dir()
    contents = [n for n in output_dir.iterdir()]
    if len(contents) > 0:
        raise RuntimeError(f"{output_dir} is not empty")

    if not properties_dir.exists():
        properties_dir.mkdir()
    assert properties_dir.is_dir()

    output_id = 1
    data = binarize_swc(swc_path)
    with open(output_dir / str(output_id), 'wb') as out_file:
        out_file.write(data)

    params = dict()
    params['@type'] = 'neuroglancer_skeletons'
    transform = [1000, 0, 0, 0,
                 0, 1000, 0, 0,
                 0, 0, 1000, 0]
    params['transform'] = transform

    vertex_attributes = []
    vertex_attributes.append({'id': str(output_id),
                              'data_type': 'float32',
                              'num_components': 1})
    params['vertex_attributes'] = [] #vertex_attributes
    params['segment_properties'] = properties_dir.name

    info_path = output_dir / 'info'
    with open(info_path, 'w') as out_file:
        out_file.write(json.dumps(params, indent=2))

    # write properties
    properties = dict()
    properties['properties'] = [{'id': 'label', 'type': 'label',
                                 'values': ['dummy']}]
    properties['ids'] = ['1']
    params = dict()
    params['inline'] = properties
    params['@type'] = 'neuroglancer_segment_properties'
    info_path = properties_dir / 'info'
    with open(info_path, 'w') as out_file:
        out_file.write(json.dumps(params, indent=2))    

def main():
    swc_path = 'data/202206241143_upload_resampled_reconstructions_16124_1/16124_5220-X13868-Y7456_reg.swc'
    convert_swc(swc_path,
                output_dir='scratch')


if __name__ == "__main__":
    main()
