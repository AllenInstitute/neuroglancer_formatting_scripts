import json
import argparse
import pathlib
import urllib.request

def simple_id_from_csv(file_path):
    """
    Scan through a file; ignore the first line as a header;
    for each row, split on comma and test of each field can
    be converted to an int. If more than one can, through an
    error. If only one can, then concatenate that into a list
    of image_series_ids
    """
    id_list = []
    with open(file_path, 'rb') as in_file:
        header = in_file.readline().decode('utf-8-sig')
        for line in in_file:
            params = line.decode('utf-8-sig').strip().split(',')
            id_value = None
            for p in params:
                try:
                    v = int(p)
                    if id_value is not None:
                        raise RuntimeError(
                            f"\nmore than one ID in\n{line}\n")
                    id_value = v
                except ValueError:
                    pass
            if id_value is None:
                raise RuntimeError(
                    f"\ncould not find ID in\n{line}\n")
            id_list.append(id_value)
    return id_list


def get_image_series_metadata(image_series_id, passed_only=True):

    url ="http://api.brain-map.org/api/v2/data/query.json?"
    url += "criteria=model::SectionImage,rma::criteria,"
    url += f"[data_set_id$eq{image_series_id}]"

    raw_response = urllib.request.urlopen(url).readlines()
    assert len(raw_response) == 1
    response = json.loads(raw_response[0])["msg"]

    image_series_metadata = []
    for element in response:
        if passed_only and element["failed"]:
            continue
        this = dict()
        image_path = pathlib.Path(element["path"])
        this["zoom"] = image_path.name
        this["storage_directory"] = str(image_path.parent)
        for k in ("x", "y", "width", "height", "resolution"):
            this[k] = element[k]
        this["sub_image_id"] = element["id"]
        this["image_series_id"] = element["data_set_id"]
        this["specimen_tissue_index"] = element["section_number"]
        image_series_metadata.append(this)
    return image_series_metadata


def get_all_image_series_metadata(
        image_series_id_list):

    result = []
    for image_series_id in image_series_id_list:
        result += get_image_series_metadata(image_series_id)
    return result


def get_specimen_metadata(image_series_id):
    """
    add metadata to one specific instance of
    image metadata
    """

    url ="http://api.brain-map.org/api/v2/data/query.json?"
    url += "criteria=model::SectionDataSet,rma::criteria,"
    url += f"[id$eq{image_series_id}],"
    url += "rma::include,genes,plane_of_section,specimen(donor(age))"

    raw_response = urllib.request.urlopen(url).readlines()
    assert len(raw_response) == 1
    metadata = json.loads(raw_response[0])["msg"]
    assert len(metadata) == 1
    metadata = metadata[0]
    ii = metadata.pop('id')
    assert ii == image_series_id
    metadata['image_series_id'] = image_series_id
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id_file_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--clobber', default=False, action='store_true')
    args = parser.parse_args()

    id_file_path = pathlib.Path(args.id_file_path)
    if not id_file_path.is_file():
        raise RuntimeError(
            f"\n{id_file_path.resolve().absolute()}\n"
            "is not a file")

    assert args.output_path is not None
    output_path = pathlib.Path(args.output_path)
    if output_path.exists():
        if not args.clobber:
            raise RuntimeError(
                f"\n{output_path.resolve().absolute()}\n"
                f"already exists")

    final_config = dict()

    image_series_id_list = simple_id_from_csv(id_file_path)
    final_config['configs'] = get_all_image_series_metadata(
                                  image_series_id_list)

    metadata = dict()
    for image_series_id in image_series_id_list:
        metadata[image_series_id] = get_specimen_metadata(image_series_id)
    final_config['metadata'] = metadata

    with open(output_path, "w") as out_file:
        out_file.write(json.dumps(final_config,
                                  indent=2))


if __name__ == "__main__":
    main()
