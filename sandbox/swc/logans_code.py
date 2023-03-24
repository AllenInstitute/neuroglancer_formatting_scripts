# copied from https://gist.github.com/GeekLogan/e02ceae25d52cf52c577dd42d205d8bf

import ngauge
import struct


def binarize_swc(swc_path):
    n = ngauge.Neuron.from_swc(swc_path)

    output_points, output_edges = bytearray(), bytearray()

    pm = {} # point map between point and line_id
    for i,pt in enumerate(n.iter_all_points()):
        pm[pt] = i
        output_points.extend( struct.pack("<fff", pt.x, pt.y, pt.z) )
        output_edges.extend( struct.pack("<II", i, pm[pt.parent]) if pt.parent else b'' )
    
    output = struct.pack("<II", len(output_points) // 12, len(output_edges) // 8)
    output += output_points + output_edges

    #output is now a bytes object containing the packed structure
    return output
