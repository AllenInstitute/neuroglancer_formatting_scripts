import pathlib
import re
import json
import shutil
import SimpleITK
import time
import numpy as np
from precomputed_utils import clean_dir
from cloudvolume import CloudVolume
from taskqueue import LocalTaskQueue
#import igneous.task_creation as igneous_task_creation


def format_ccf_annotations(
        annotation_path,
        segmenation_path,
        output_dir,
        clobber):
    """
    annotation_path -- path to text file with label names

    segmentation_path -- path to ccf nii.gz file

    output_dir -- probably ends with ccf_annotations....?
    """
    assert annotation_path is not None
    annotation_path = pathlib.Path(args.annotation_path)
    assert annotation_path.is_file()

    assert segmentation_path is not None
    segmentation_path = pathlib.Path(args.segmentation_path)
    assert segmentation_path.is_file()

    assert output_dir is not None
    output_dir = pathlib.Path(args.output_dir)
    if output_dir.exists():
        if not clobber:
            raise RuntimeError(f"{output_dir} exists")
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    assert output_dir.is_dir()

    write_labels_info_file(
        annotation_path=annotation_path,
        output_dir=output_dir)

    process_segmentation(
        segmentation_path=segmentation_path,
        layer_dir=output_dir)

    print("Successfully formatted CCF annotations!")

def get_labels(annotation_path):
    name_lookup = {}
    name_set = set()
    name_pattern = re.compile('".*"')
    with open(annotation_path, 'r') as in_file:
        for line in in_file:
            idx = int(line.split()[0])
            name = name_pattern.findall(line)[0]
            name = name.replace('"','')
            name = name.split(' - ')[0]
            if name in name_set:
                raise RuntimeError(
                    f"{name} repeated")
            if idx in name_lookup:
                raise RuntimeError(
                    f"idx {idx} repeated")
            name_set.add(name)
            name_lookup[idx] = name
    return name_lookup


def format_labels(labels):
    """
    convert an idx -> label lookup into a dict conforming to the metadata
    schema expected by neuroglancer for a segmentation layer
    """
    output = dict()
    output["@type"] = "neuroglancer_segment_properties"
    inline = dict()
    inline["ids"] = []
    properties = dict()
    properties["id"] = "label"
    properties["type"] = "label"
    properties["values"] = []

    k_list = list(labels.keys())
    k_list.sort()
    for k in k_list:
        value = labels[k]
        properties["values"].append(value)
        inline["ids"].append(str(k))

    inline["properties"] = [properties]
    output["inline"] = inline
    return output


def make_info_file(
        resolution_xyz,
        volume_size_xyz,
        layer_dir):
    """
    Shamelessly copied from

    https://github.com/PrincetonUniversity/lightsheet_helper_scripts/blob/master/neuroglancer/brodylab_MRI_atlas_customizations.ipynb

    Make an JSON-formatted file called the "info" file
    for use with the precomputed data format. 
    Precomputed is one of the formats that Neuroglancer can read in.  
    --- parameters ---
    resolution_xyz:      A tuple representing the size of the pixels (dx,dy,dz) 
                         in nanometers, e.g. (20000,20000,5000) for 20 micron x 20 micron x 5 micron
    
    volume_size_xyz:     A tuple representing the number of pixels in each dimension (Nx,Ny,Nz)

                         
    layer_dir:           The directory where the precomputed data will be
                         saved
    """
    info = CloudVolume.create_new_info(
        num_channels = 1,
        layer_type = 'segmentation', # 'image' or 'segmentation'
        data_type = 'uint16', # 32-bit not necessary for atlases unless you have > 2^(32)-1 labels. Use smallest possible  
        encoding = 'raw', # other options: 'jpeg', 'compressed_segmentation' (req. uint32 or uint64)
        resolution = resolution_xyz, # X,Y,Z values in nanometers, 40 microns in each dim
        voxel_offset = [ 0, 0, 0 ], # values X,Y,Z values in voxels
        chunk_size = [ 1024, 1024, 1 ], # rechunk of image X,Y,Z in voxels.
        volume_size = volume_size_xyz, # X,Y,Z size in voxels
    )

    vol = CloudVolume(f'file://{layer_dir}', info=info)
    vol.provenance.description = "A test info file" # can change this if you want a description
    vol.provenance.owners = [''] # list of contact email addresses
    # Saves the info and provenance files for the first time
    vol.commit_info() # generates file://bucket/dataset/layer/info json file
    vol.commit_provenance() # generates file://bucket/dataset/layer/provenance json file
    # add a key for the segment properties that points to the directory that holds the segment properties info file
    info_dict = vol.info
    info_dict['segment_properties'] = "segment_properties"
    info_filename = '/'.join(vol.info_cloudpath.split('/')[2:]) 
    with open(info_filename,'w') as outfile:
        json.dump(info_dict,outfile,sort_keys=True,indent=2)
    print("Created CloudVolume info file: ",vol.info_cloudpath)

    return vol


# define the meshing function
def make_3d_mesh(vol,n_cores):
    """
    Shamelessly copied from

    https://github.com/PrincetonUniversity/lightsheet_helper_scripts/blob/master/neuroglancer/brodylab_MRI_atlas_customizations.ipynb

    This function makes the mesh so that when you
    load your segmentation layer into neuroglancer, you 
    will be able to see whichever segments are highlighted
    in the 3d viewer as well as all of the other panels. 

    It uses parallel processing to speed up the meshing
    ---parameters---
    vol:     The cloudvolume object
    n_cores: Number of cores to use
    """
    # Mesh using the  cores, use True to use all cores
    cloudpath = vol.cloudpath
    with LocalTaskQueue(parallel=n_cores) as tq:
        tasks = igneous_task_creation.create_meshing_tasks(
                        cloudpath,
                        mip=0,
                        shape=(256, 256, 256))
        tq.insert_all(tasks)
        tasks = igneous_task_creation.create_mesh_manifest_tasks(cloudpath)
        tq.insert_all(tasks)
    print("Done!")


def process_segmentation(
        segmentation_path,
        layer_dir):
    """
    Save the atlas to layer_dir
    """

    img = SimpleITK.ReadImage(segmentation_path)
    arr = SimpleITK.GetArrayFromImage(img).transpose(2, 1, 0)
    arr = np.round(arr).astype(np.uint16)

    mm_to_nm = 10**6

    resolution_xyz = (int(mm_to_nm*float(img.GetMetaData('pixdim[1]'))),
                      int(mm_to_nm*float(img.GetMetaData('pixdim[2]'))),
                      int(mm_to_nm*float(img.GetMetaData('pixdim[3]'))))

    cloud_volume = make_info_file(
        resolution_xyz=resolution_xyz,
        volume_size_xyz=arr.shape,
        layer_dir=layer_dir)

    print(cloud_volume.shape)
    print(arr.shape)

    # the example notebook uses multiprocessing to load one
    # chunk in z at a time; not sure why. Maybe CloudVolume is
    # expensive under the hood?
    #
    # these preliminary atlases are small enough that it does not
    # matter
    t0 = time.time()
    cloud_volume[:, :, :, 0] = arr
    print(f"loaded in {time.time()-t0:.2e}")

    # for some reason 3D rendering does not work, unless I serve
    # the data through CloudVolume.viewer
    # This is something to investigate before we move on
    # to morphology data
    #
    #t0 = time.time()
    #make_3d_mesh(cloud_volume, 8)
    #print(f"made 3D mesh in {time.time()-t0:.2e}")

    clean_dir(layer_dir)

def write_labels_info_file(
        annotation_path,
        output_dir):

    labels = get_labels(annotation_path)
    info = format_labels(labels)
    new_dir = output_dir / "segment_properties"
    new_dir.mkdir()
    info_path = new_dir / "info"
    with open(info_path, "w") as out_file:
        out_file.write(json.dumps(info, indent=2))
