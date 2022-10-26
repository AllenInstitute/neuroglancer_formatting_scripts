import pathlib

def sanitize_cluster_name(name):
    for bad_char in (' ', '/'):
        name = name.replace(bad_char, '_')
    return name

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
