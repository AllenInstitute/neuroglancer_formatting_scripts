# In this module, I will implement the compressed_segmentation framework
# defined for neuroglancer here
# https://github.com/google/neuroglancer/tree/master/src/neuroglancer/sliceview/compressed_segmentation

import numpy as np


def compress_ccf_data(
        data,
        file_path,
        blocksize=64):
    """
    Write data (a np.ndarray) to file_path, compressing according
    to neuroglancer's compressed_segmentation framework using the
    specified blocksize.
    """

    nx = data.shape[0]
    ny = data.shape[1]
    nz = data.shape[2]

    n_tot = (nz/blocksize)*(ny/blocksize)*(nx/blocksize)

    encoding_list = []
    ct = 0
    for z0 in range(0, nz, blocksize):
        z1 = min(z0+blocksize, nz)
        for y0 in range(0, ny, blocksize):
            y1 = min(y0+blocksize, ny)
            for x0 in range(0, nx, blocksize):
                x1 = min(x0+blocksize, nx)
                block = get_block(
                    all_data=data,
                    x_spec=(x0, x1),
                    y_spec=(y0, y1),
                    z_spec=(z0, z1),
                    blocksize=blocksize)

                if block.shape != (blocksize, blocksize, blocksize):
                    raise RuntimeError(
                        f"block.shape {block.shape} but "
                        f"blocksize = {blocksize}")

                encoding_list.append(encode_block(block))
                ct += 1

    # data will be orderd as recommended
    # * headers
    # * data for block0
    # * lookup for block0
    # * data for block1
    # * lookup for block1
    n_blocks = len(encoding_list)
    header_offset = n_blocks*2
    header_list = []
    running_offset = header_offset
    for i_block in range(n_blocks):
        this_encoding = encoding_list[i_block]

        n_bits = this_encoding['n_bits']

        n_lookup_bytes = len(this_encoding['lookup_table'])
        assert n_lookup_bytes % 4 == 0
        n_lookup = n_lookup_bytes // 4

        n_data_bytes = len(this_encoding['encoded_data'])
        assert n_data_bytes % 4 == 0
        n_data = n_data_bytes // 4

        data_offset = running_offset
        assert data_offset < 2**32
        running_offset += n_data

        lookup_offset = running_offset
        assert lookup_offset < 2**24
        running_offset += n_lookup

        this_header = b''
        this_header += lookup_offset.to_bytes(3, byteorder='little')
        this_header += n_bits.to_bytes(1, byteorder='little')
        this_header += data_offset.to_bytes(4, byteorder='little')
        if len(this_header) != 8:
            raise RuntimeError(
                f"header\n{this_header}\nlen {len(this_header)}")
        header_list.append(this_header)

    with open(file_path, 'wb') as out_file:
        # specify that this is just one channel
        out_file.write((1).to_bytes(4, byteorder='little'))
        for header in header_list:
            out_file.write(header)
        for encoding in encoding_list:
            out_file.write(encoding['encoded_data'])
            out_file.write(encoding['lookup_table'])


def get_block(
        all_data,
        x_spec,
        y_spec,
        z_spec,
        blocksize):
    """
    Get and return a block of data from all_data (a np.ndarray).

    x_spec, y_spec, and z_spec are of the form (min_val, max_val)
    and specify the block of data to return.

    blocksize is the desired size (the block will be a cube)
    of the datablock. if x_spec, y_spec, z_spec specify a block
    that is smaller than a (blocksize, blocksize, blocksize) cube,
    use np.pad to fill out the block.
    """

    block = all_data[x_spec[0]:x_spec[1],
                     y_spec[0]:y_spec[1],
                     z_spec[0]:z_spec[1]]

    pad_x = 0
    if block.shape[0] != blocksize:
        pad_x = blocksize-block.shape[0]
    pad_y = 0
    if block.shape[1] != blocksize:
        pad_y = blocksize-block.shape[1]
    pad_z = 0
    if block.shape[2] != blocksize:
        pad_z = blocksize-block.shape[2]

    if pad_x + pad_y + pad_z > 0:
        val = block[0,0,0]
        block = np.pad(block,
                       pad_width=[[0, pad_x],
                                  [0, pad_y],
                                  [0, pad_z]],
                       mode='constant',
                       constant_values=val)
    return block


def encode_block(data):
    """
    Returns dict containing

    the byte stream that is the encoded data

    the byte stream that is the lookup table of values

    the number of bits used to encode the values in the
    lookup table
    """
    encoding = get_block_lookup_table(data)

    n_bits = encoding['n_bits_to_encode']
    encoder = encoding['dict']

    nx = data.shape[0]
    ny = data.shape[1]
    nz = data.shape[2]

    byte_stream = b''
    ct = data.size
    if n_bits > 0:
        bit_stream = block_to_bits(
                        block=data,
                        encoder_dict=encoder,
                        n_bits=n_bits)

        byte_stream = bits_to_bytes(bit_stream)

    expected_len = np.ceil(ct*n_bits/8).astype(int)
    if len(byte_stream) != expected_len:
        raise RuntimeError(
            f"len bytes {len(byte_stream)}\n"
            f"expected {expected_len}\n"
            f"{(nx, ny, nz)}\n"
            f"{n_bits}")

    return {'encoded_data': byte_stream,
            'lookup_table': encoding['bytes'],
            'n_bits': n_bits}

def block_to_bits(block, encoder_dict, n_bits):
    """
    Convert block into a string of bits encoded according
    to encoder_dict.

    Parameters
    ----------
    block: np.ndarray
        Data to encode

    encoder_dict: dict
        Dict mapping values in block to encoded values
        (smaller ints)

    n_bits: int
        Number of bits used to store each value in the
        returned bit stream

    Returns
    -------
    bit_stream: np.ndarray
       Booleans representing the bits of the encoded
       values. Should be block.size*n_bits long.
       Values will be little-endian (least significatn
       bit first).
    """
    n_total_bits = block.size*n_bits
    if n_total_bits % 32 > 0:
        n_total_bits += (32-(n_total_bits%32))
    assert n_total_bits%32 == 0

    bit_stream = np.zeros(n_total_bits, dtype=bool)
    bit_masks = np.array([2**ii for ii in range(n_bits)]).astype(int)
    block = np.array([encoder_dict[val] for val in block.flatten('F')])

    for i_bit in range(n_bits):
        detections = block & bit_masks[i_bit]
        bit_stream[i_bit::n_bits] = (detections > 0)

    return bit_stream

def bits_to_bytes(bit_stream):
    """
    Convert a bit stream (as produced by block_to_bits) to
    a byte stream that can be written out to the compressed
    data file.
    """

    n_bits = len(bit_stream)
    assert n_bits % 32 == 0

    # Convert the bit stream into a series of little-endian
    # 32 bit unsigned integers. These values will ultimately
    # get converted to bytes and stored in the output byte
    # stream.
    bit_grid = np.array(bit_stream).reshape(n_bits//32, 32)
    values = np.zeros(bit_grid.shape[0], dtype=np.uint32)
    pwr = 1
    for icol in range(bit_grid.shape[1]):
        these_bits = bit_grid[:, icol]
        values[these_bits] += pwr
        pwr *= 2

    # initialize empty byte stream
    byte_stream = bytearray(n_bits//8)

    # transcribe values in byte stream
    for i_val, val in enumerate(values):
        byte_stream[i_val*4:(i_val+1)*4] = int(val).to_bytes(4, byteorder='little', signed=False)

    return bytes(byte_stream)


def get_block_lookup_table(data):
    """
    Get the lookup table for encoded values in data.

    Returns
    -------
    dict mapping raw values to encoded values

    byte stream representing the lookup table of raw values
    (this is just a sequence of values; the value's position
    in bytestream represents its encoded value, i.e. the 5th
    raw value in byte stream gets encoded to the value 5)

    number of bits used to encode each value
    """
    max_val = data.max()
    if data.max() >= 2**32:
        raise RuntimeError(f"max_val {max_val} >= 2**32")

    unq_values = np.unique(data).astype(np.uint32)
    n_unq = len(unq_values)
    raw_n_bits_to_encode = np.ceil(np.log(n_unq)/np.log(2)).astype(int)

    if raw_n_bits_to_encode == 0:
        n_bits_to_encode = 0
    else:
        n_bits_to_encode = 1
        while n_bits_to_encode < raw_n_bits_to_encode:
            n_bits_to_encode *= 2

    if n_bits_to_encode >= 32:
        raise RuntimeError(
            f"n_bits_to_encode {n_bits_to_encode}\n"
            f"n_unq {n_unq}")

    val_to_encoded = dict()
    byte_stream = b''

    for ii, val in enumerate(unq_values):
        val_to_encoded[val] = ii

        # bytestream will be encoded_to_val
        # since that is used for *decoding* the data
        val_bytes = int(val).to_bytes(4, byteorder='little')

        #encoded_bytes = int(ii).to_bytes(4, byteorder='little')
        #byte_stream += encoded_bytes

        byte_stream += val_bytes

    assert len(byte_stream) == 4*len(unq_values)

    return {'bytes': byte_stream,
            'dict': val_to_encoded,
            'n_bits_to_encode': n_bits_to_encode}
