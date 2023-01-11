# In this module, I will implement the compressed_segmentation framework
# defined for neuroglancer here
# https://github.com/google/neuroglancer/tree/master/src/neuroglancer/sliceview/compressed_segmentation

import numpy as np


def compress_ccf_data(
        data,
        file_path,
        blocksize=64):

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
    print('writing data')
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

    print(f"writing compressed data to {file_path}")
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
    current_int = 0
    bit_count = 0

    ct = data.size
    if n_bits > 0:
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    idx = (ix+nx*(iy+ny*iz))

                    val = data[ix, iy, iz]

                    encoded_val = encoder[val]
                    ct += 1
                    (current_int,
                     bit_count,
                     byte_stream) = update_byte_stream(
                         this_value=encoded_val,
                         n_bits=n_bits,
                         current_int=current_int,
                         bit_count=bit_count,
                         byte_stream=byte_stream)

        if bit_count > 0:
            byte_stream = add_int_to_byte_stream(
                    current_int=current_int,
                    byte_stream=byte_stream)

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


def update_byte_stream(
        this_value,
        n_bits,
        current_int,
        bit_count,
        byte_stream):
    """
    this_value is the value being added to the byte_stream

    n_bits is the number of bits used to encode a value

    current_int is the cache of the integer being written out
    to byte_stream

    bit_count is where we are in the current_byte

    byte_stream is the byte_stream being updated

    Returns
    -------
    updated values for current_int, bit_count, byte_stream
    """

    # is this adding the bits in the correct order...
    # I think so; the least significant bit comes first,
    # (unless it should not...)

    assert this_value < 2**n_bits

    # reverse it so the number is little endian
    this_binary = f'{this_value:0{n_bits}b}'[::-1]
    if n_bits > 0:
        if len(this_binary) != n_bits:
            raise RuntimeError(
                f"value {this_value}\nbits {n_bits}\n"
                f"binary {this_binary}")
    else:
        assert this_value == 0

    pwr = 2**bit_count
    for idx in range(n_bits):
        if this_binary[idx] == '1':
            current_int += pwr
        pwr *= 2
        bit_count += 1
        if bit_count == 32:
            byte_stream = add_int_to_byte_stream(
                    current_int=current_int,
                    byte_stream=byte_stream)
            bit_count = 0
            current_int = 0

    return (current_int,
            bit_count,
            byte_stream)


def add_int_to_byte_stream(
        current_int,
        byte_stream):
    """
    Add current_int as a 32 bit little-endian integer to
    byte_stream
    """
    n0 = len(byte_stream)
    assert current_int < 2**32
    as_bytes = int(current_int).to_bytes(4, byteorder='little')
    byte_stream += as_bytes
    assert len(byte_stream) == (n0+4)
    return byte_stream


def get_block_lookup_table(data):
    """
    Get the lookup table for encoded values in data.

    Return both as a string of bytes and as a python dictionary
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
