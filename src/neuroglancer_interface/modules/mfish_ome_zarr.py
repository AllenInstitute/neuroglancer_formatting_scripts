import pathlib
import multiprocessing
from neuroglancer_interface.utils.data_utils import (
    write_nii_file_list_to_ome_zarr)
from neuroglancer_interface.classes.metadata_collectors import (
    CellTypeMetadataCollector)
from neuroglancer_interface.utils.mfish_utils import (
    gene_from_fname)


def convert_mfish_to_ome_zarr(
        input_dir: str,
        output_dir: str,
        clobber: bool,
        downscale: int,
        n_processors: int,
        structure_set_masks=None,
        structure_masks=None,
        n_test=None):
    """
    input_dir -- where the gene.nii.gz files live

    output_dir -- the directory where ome zarr will be
    written (e.g. mouse_5/mfish_heatmaps)

    clobber -- whether or not to overwrite data; should probably
    always be False in bundle runner

    downscale -- factor by which to downscale images at each step

    n_processors -- number of independent workers
    """

    output_dir = pathlib.Path(output_dir)
    input_dir = pathlib.Path(input_dir)
    if not input_dir.is_dir():
        raise RuntimeError(
            "In convert_mfish_to_ome_zarr, input_dir\n"
            f"{input_dir.resolve().absolute()}\n"
            "is not a dir")

    metadata_path = output_dir / 'metadata.json'

    metadata_collector = CellTypeMetadataCollector(
            metadata_output_path=metadata_path,
            structure_set_masks=structure_set_masks,
            structure_masks=structure_masks)

    mgr = multiprocessing.Manager()
    metadata_collector.set_lock(mgr.Lock())
    metadata_collector.metadata = mgr.dict()

    fname_list = [n for n in input_dir.rglob('*nii.gz')]

    if n_test is not None:
        fname_list = fname_list[:n_test]

    fname_list.sort()

    genes_loaded = set()
    gene_list = []
    for fname in fname_list:
        gene = gene_from_fname(fname)
        if gene in genes_loaded:
            msg = f"{gene} appears twice\n"
            msg += f"{fname}\n"
            raise RuntimeError(msg)
        gene_list.append(gene)

    root_group = write_nii_file_list_to_ome_zarr(
        file_path_list=fname_list,
        group_name_list=gene_list,
        output_dir=output_dir,
        downscale=downscale,
        clobber=clobber,
        n_processors=n_processors,
        metadata_collector=metadata_collector)


    metadata_collector.write_to_file()
