import pathlib
import json
from logans_code import binarize_swc


def convert_swc(
        swc_path_list,
        output_dir):

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


    params = dict()
    params['@type'] = 'neuroglancer_skeletons'
    transform = [1000, 0, 0, 0,
                 0, 1000, 0, 0,
                 0, 0, 1000, 0]
    params['transform'] = transform

    vertex_attributes = []
    n_good = 0
    n_bad = 0
    id_to_label = dict()
    for output_id, swc_path in enumerate(swc_path_list):

        parent_dir = swc_path.parent.name
        this_label = f'{parent_dir}/{swc_path.name}'

        swc_path = pathlib.Path(swc_path)

        if not swc_path.is_file():
            raise RuntimeError(f"{swc_file} is not file")

        try:
            data = binarize_swc(swc_path)
            with open(output_dir / str(output_id), 'wb') as out_file:
                out_file.write(data)

            vertex_attributes.append({'id': str(output_id),
                                      'data_type': 'float32',
                                      'num_components': 1})

            id_to_label[str(output_id)] = this_label

            n_good += 1
        except:
            n_bad += 1

            print(f"problem with {swc_path} -- {n_good} {n_bad}")

        continue

    print(f"n_good {n_good} n_bad {n_bad}")

    params['vertex_attributes'] = [] #vertex_attributes
    params['segment_properties'] = properties_dir.name

    info_path = output_dir / 'info'
    with open(info_path, 'w') as out_file:
        out_file.write(json.dumps(params, indent=2))

    # write properties
    properties = dict()

    id_list = []
    label_list = []
    for k in id_to_label:
        id_list.append(k)
        label_list.append(id_to_label[k])

    properties['properties'] = [{'id': 'label', 'type': 'label',
                                 'values': label_list}]
    properties['ids'] = id_list
    params = dict()
    params['inline'] = properties
    params['@type'] = 'neuroglancer_segment_properties'
    info_path = properties_dir / 'info'
    with open(info_path, 'w') as out_file:
        out_file.write(json.dumps(params, indent=2))    

def main():

    data_dir = pathlib.Path('data')
    swc_path_list = [n for n in data_dir.rglob('**/*.swc')]

    convert_swc(swc_path_list,
                output_dir='scratch/swc')


if __name__ == "__main__":
    main()
