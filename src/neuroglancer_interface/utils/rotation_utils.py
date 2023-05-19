import numpy as np

def get_rotation_matrix(
        aa, bb, cc, dd):
    """
    Convert quaternion terms into a rotation matrix

    See:
    https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/quatern.html
    """
    if not np.allclose(aa**2+bb**2+cc**2+dd**2, 1.0):
        raise RuntimeError(
            f"quaternion terms {aa}, {bb}, {cc}, {dd} "
            "do not sum to one in quadrature")

    rot = np.zeros((3,3), dtype=float)
    rot[0, 0] = aa**2+bb**2-cc**2-dd**2
    rot[1, 1] = aa**2+cc**2-bb**2-dd**2
    rot[2, 2] = aa**2+dd**2-bb**2-cc**2

    #rot[0, 1] = 2*bb*cc-2*aa*dd
    #rot[0, 2] = 2*bb*dd+2*aa*cc
    #rot[1, 0] = 2*bb*cc+2*aa*dd
    #rot[1, 2] = 2*cc*dd-2*aa*bb
    #rot[2, 0] = 2*bb*dd-2*aa*cc
    #rot[2, 1] = 2*cc*dd+2*aa*bb

    # actually https://www.mathworks.com/help/aeroblks/quaternionstodirectioncosinematrix.html
    rot[0,1] = 2*(bb*cc+aa*dd)
    rot[0,2] = 2*(bb*dd-aa*cc)
    rot[1,0] = 2*(bb*cc-aa*dd)
    rot[1,2] = 2*(cc*dd+aa*bb)
    rot[2,0] = 2*(bb*dd+aa*cc)
    rot[2,1] = 2*(cc*dd-aa*bb)

    return rot

def get_coord_mesh(data_shape):
    """
    Return a 3xN array encoding the 3D matrix coordinates
    of voxels in a data array with N total voxels
    """

    max_d_log2 = np.log2(max(data_shape))
    if max_d_log2 < 8:
        mesh_dtype = np.uint8
    elif max_d_log2 < 16:
        mesh_dtype = np.uint16
    elif max_d_log2 < 32:
        mesh_dtype = np.uint32
    else:
        mesh_dtype = np.uint

    (xx_mesh,
     yy_mesh,
     zz_mesh) = np.meshgrid(
                    np.arange(data_shape[0], dtype=mesh_dtype),
                    np.arange(data_shape[1], dtype=mesh_dtype),
                    np.arange(data_shape[2], dtype=mesh_dtype))

    xx_mesh = xx_mesh.flatten()
    yy_mesh = yy_mesh.flatten()
    zz_mesh = zz_mesh.flatten()
    coord_mesh = np.vstack([xx_mesh, yy_mesh, zz_mesh])
    assert coord_mesh.shape == (3, data_shape[0]*data_shape[1]*data_shape[2])
    return coord_mesh


def rotate_matrix(
        data,
        rotation_matrix):
    """
    rotate the data according to the specified rotation matrix
    """
    data_idx = get_coord_mesh(data.shape)
    rotated_idx = np.dot(np.array(rotation_matrix).astype(np.float32),
                         data_idx)

    for ix in range(3):
        rotated_idx[ix, :] -= rotated_idx[ix, :].min()

    this_i_info = np.iinfo(data_idx.dtype)
    if rotated_idx.min() < 0:
        raise RuntimeError("got negative minimum index")
    if rotated_idx.max() > this_i_info.max:
        raise RuntimeError("max index too great")
    rotated_idx = np.round(rotated_idx).astype(data_idx.dtype)

    new_shape = (rotated_idx[0].max()+1,
                 rotated_idx[1].max()+1,
                 rotated_idx[2].max()+1)

    for v in data.shape:
        assert v in new_shape

    new_data = np.zeros(new_shape, dtype=data.dtype)
    new_data[rotated_idx[0, :],
             rotated_idx[1, :],
             rotated_idx[2, :]] = data[data_idx[0, :],
                                       data_idx[1, :],
                                       data_idx[2, :]]

    return new_data
