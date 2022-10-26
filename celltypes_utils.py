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

    with open(anno_path, "r") as in_file:
        header = in_file.readline()
        for line in in_file:
            params = line.replace('"', '').strip().split(',')
            assert len(params) == 4
            cluster_name = sanitize_cluster_name(params[1])
            valid_clusters.add(cluster_name)
            subclass_name = sanitize_cluster_name(params[2])
            class_name = sanitize_cluster_name(params[3])

            if subclass_name not in subclass_to_clusters:
                subclass_to_clusters[subclass_name] = []
            if class_name not in class_to_clusters:
                class_to_clusters[class_name] = []

            subclass_to_clusters[subclass_name].append(cluster_name)
            class_to_clusters[class_name].append(cluster_name)

    return subclass_to_clusters, class_to_clusters, valid_clusters
