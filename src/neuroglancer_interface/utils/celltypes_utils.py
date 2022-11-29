import pathlib



def read_manifest(manifest_path):
    """
    Get a lookup table from filename to
    celltype name and machine readable group
    name from the manifest.csv files written
    by Lydia's script
    """
    label_idx = None
    path_idx = None
    with open(manifest_path, "r") as in_file:
        header = in_file.readline().strip().split(',')
        for idx, val in enumerate(header):
            if val == 'label':
                label_idx = idx
            elif val == 'file_name':
                path_idx = idx
        assert label_idx is not None
        assert path_idx is not None
        file_path_list = []
        human_readable_list = []
        for line in in_file:
            line = line.strip().split(',')
            pth = line[path_idx]
            human_readable = line[label_idx]
            file_path_list.append(pth)
            human_readable_list.append(human_readable)

    (sanitized_list,
     _ ) = sanitize_cluster_name_list(human_readable_list)

    result = dict()
    for file_path, human_readable, sanitized in zip(file_path_list,
                                                    human_readable_list,
                                                    sanitized_list):
        result[file_path] = {"human_readable": human_readable,
                             "machine_readable": sanitized}

    return result


def sanitize_cluster_name(name):
    for bad_char in (' ', '/'):
        name = name.replace(bad_char, '_')
    return name


def sanitize_cluster_name_list(
        raw_cluster_name_list):
    sanitized_name_set = set()
    sanitized_name_list = []
    desanitizer = dict()
    for name in raw_cluster_name_list:
        sanitized_name = sanitize_cluster_name(name)
        if name in sanitized_name_set:
            raise RuntimeError(
                    f"{sanitized_name} occurs more than once")
        sanitized_name_set.add(sanitized_name)
        sanitized_name_list.append(sanitized_name)
        desanitizer[sanitized_name] = name
    return sanitized_name_list, desanitizer


def get_class_lookup(
        anno_path):
    """
    returns subclass_to_clusters and class_to_clusters which
    map the names of classes to lists of the names of clusters
    therein

    also return a set containing all of the valid cluster names
    """

    anno_path = pathlib.Path(anno_path)
    if not anno_path.is_file():
        raise RuntimeError(f"{anno_path} is not a file")

    subclass_to_clusters = dict()
    class_to_clusters = dict()
    valid_clusters = set()

    desanitizer = dict()

    with open(anno_path, "r") as in_file:
        header = in_file.readline()
        for line in in_file:
            params = line.replace('"', '').strip().split(',')
            assert len(params) == 4
            cluster_name = params[1]
            subclass_name = params[2]
            class_name = params[3]

            sanitized_cluster_name = sanitize_cluster_name(cluster_name)
            sanitized_subclass_name = sanitize_cluster_name(subclass_name)
            sanitized_class_name = sanitize_cluster_name(class_name)

            for dirty, clean in zip((cluster_name,
                                     subclass_name,
                                     class_name),
                                    (sanitized_cluster_name,
                                     sanitized_subclass_name,
                                     sanitized_class_name)):
                if clean in desanitizer:
                    if desanitizer[clean] != dirty:
                        msg = "\nmore than one way to desanitize "
                        msg += f"{clean}\n"
                        msg += f"{dirty}\n"
                        msg += f"{desanitizer[clean]}\n"
                        raise RuntimeError(msg)
                desanitizer[clean] = dirty

            valid_clusters.add(sanitized_cluster_name)

            if subclass_name not in subclass_to_clusters:
                subclass_to_clusters[sanitized_subclass_name] = []
            if class_name not in class_to_clusters:
                class_to_clusters[sanitized_class_name] = []

            subclass_to_clusters[sanitized_subclass_name].append(
                sanitized_cluster_name)

            class_to_clusters[sanitized_class_name].append(
                sanitized_cluster_name)

    return (subclass_to_clusters,
            class_to_clusters,
            valid_clusters,
            desanitizer)
