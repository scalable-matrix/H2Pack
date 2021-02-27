#!/usr/bin/env python3

import json
import sys
import struct

class EmptyClass(object):
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=2)

def hex_to_double(f):
    return struct.unpack('!d', bytes.fromhex(f))[0]

def metadata_txt_to_json(meta_txt_fname, meta_json_fname, aux_json_fname):
    txt_file = open(meta_txt_fname, 'r')
    lines = txt_file.readlines()
    txt_file.close()

    meta_json = EmptyClass()   # Metadata JSON
    aux_json  = EmptyClass()   # Auxiliary JSON

    # 1. Metadata: H2 / HSS common part
    aux_json.dim_point                  = int(lines[0])     # C.1 dim_point
    aux_json.dim_kernel                 = int(lines[1])     # C.2 dim_kernel
    aux_json.num_point                  = int(lines[2])     # C.3 num_point
    meta_json.nrow_matrix               = int(lines[3])     # A.1 nrow_matrix
    meta_json.ncol_matrix               = int(lines[4])     # A.2 ncol_matrix
    meta_json.is_symmetric              = int(lines[5])     # A.3 is_symmetric
    meta_json.num_node_row              = int(lines[6])     # A.4 num_node_row
    meta_json.num_node_col              = int(lines[7])     # A.5 num_node_col
    meta_json.root_node_row             = int(lines[8])     # A.6 root_node_row
    meta_json.root_node_col             = int(lines[9])     # A.7 root_node_col
    meta_json.num_level_row             = int(lines[10])    # A.8 num_level_row
    meta_json.num_level_col             = int(lines[11])    # A.9 num_level_col
    aux_json.is_HSS                     = int(lines[12])    # C.4 is_HSS
    aux_json.min_adm_level              = int(lines[13])    # C.5 min_adm_level
    meta_json.num_inadmissible_blocks   = int(lines[14])    # A.14 num_inadmissible_blocks - n_leaf_node
    meta_json.num_admissible_blocks     = int(lines[15])    # A.15 num_admissible_blocks
    meta_json.has_partial_adm_blocks    = int(lines[16])    # A.16 has_partial_adm_blocks
    curr_row = 17

    # 2. Metadata: partitioning tree
    # A.10 nodes_row; A.11 nodes_col == NULL since H2 matrix is symmetric
    nodes_row = []
    num_leaf_node = 0
    for i in range(meta_json.num_node_row):
        raw_data = [x for x in lines[curr_row + i].split(' ') if x]
        node_i = EmptyClass()
        node_i.index        = int(raw_data[0])  # A.10.1 index
        node_i.level        = int(raw_data[1])  # A.10.2 level
        node_i.cluster_head = int(raw_data[2])  # A.10.3 cluster_head
        node_i.cluster_tail = int(raw_data[3])  # A.10.4 cluster_tail
        node_i.num_children = int(raw_data[4])  # A.10.5 num_children
        if 0 == node_i.num_children:
            num_leaf_node += 1
        # A.10.6 children
        node_i.children = []
        for j in range(node_i.num_children):
            node_i.children.append(int(raw_data[5 + j]))
        nodes_row.append(node_i)
    meta_json.nodes_row = nodes_row
    curr_row += meta_json.num_node_row
    meta_json.num_inadmissible_blocks += num_leaf_node

    # 3. Metadata data: U matrices
    # A.12 basis_matrices_row (A.13 ignored since H2 matrix is symmetric)
    U_mat = []
    for i in range(meta_json.num_node_row):
        raw_data = [x for x in lines[curr_row + i].split(' ') if x]
        U_i = EmptyClass()
        U_i.node    = int(raw_data[0])  # A.12.1 node
        U_i.num_row = int(raw_data[1])  # A.12.2 num_row
        U_i.num_col = int(raw_data[2])  # A.12.3 num_col
        U_mat.append(U_i)
    meta_json.basis_matrices_row = U_mat
    curr_row += meta_json.num_node_row

    # 4. Metadata data: B matrices
    B_mat = []
    for i in range(meta_json.num_admissible_blocks):
        raw_data = [x for x in lines[curr_row + i].split(' ') if x]
        B_i = EmptyClass()
        B_i.node_row    = int(raw_data[0])  # A.17.1 node_row
        B_i.node_col    = int(raw_data[1])  # A.17.2 node_col
        B_i.num_row     = int(raw_data[2])  # A.17.3 num_row
        B_i.num_col     = int(raw_data[3])  # A.17.4 num_col
        B_i.is_part_adm = int(raw_data[4])  # A.17.5 is_part_adm
        B_mat.append(B_i)
    meta_json.B_matrices = B_mat
    curr_row += meta_json.num_admissible_blocks

    # 5. Metadata data: D matrices
    D_mat = []
    for i in range(meta_json.num_inadmissible_blocks):
        raw_data = [x for x in lines[curr_row + i].split(' ') if x]
        D_i = EmptyClass()
        D_i.node_row = int(raw_data[0]) # A.18.1 node_row
        D_i.node_col = int(raw_data[1]) # A.18.2 node_col
        D_i.num_row  = int(raw_data[2]) # A.18.3 num_row
        D_i.num_col  = int(raw_data[3]) # A.18.4 num_col
        D_mat.append(D_i)
    meta_json.D_matrices = D_mat
    curr_row += meta_json.num_inadmissible_blocks

    # 6. Other necessary information for H2Pack
    aux_json.max_leaf_points     = int(lines[curr_row])         # C.6 max_leaf_points
    aux_json.QR_stop_tol         = float(lines[curr_row + 1])   # C.7 QR_stop_tol
    aux_json.has_skeleton_points = int(lines[curr_row + 2])     # C.8 has_skeleton_points
    curr_row += 3
    # C.9 point_coordinate
    # Cast it from uint64_t back to double
    coord = []
    for i in range(aux_json.num_point):
        raw_data = [x for x in lines[curr_row + i].split(' ') if x]
        for j in range(aux_json.dim_point):
            coord.append(hex_to_double(raw_data[j]))
    aux_json.point_coordinate = coord
    curr_row += aux_json.num_point
    # C.10 permutation_array
    perm = []
    for i in range(aux_json.num_point):
        perm.append(int(lines[curr_row + i]))
    aux_json.permutation_array = perm
    curr_row += aux_json.num_point
    # C.11 skeleton_point
    node_skel = []
    for i in range(meta_json.num_node_row):
        raw_data = [x for x in lines[curr_row + i].split(' ') if x]
        skel_i = EmptyClass()
        skel_i.node = int(raw_data[0])
        skel_i.num_skeleton_point = int(raw_data[1])
        pt_idx = []
        for j in range(skel_i.num_skeleton_point):
            pt_idx.append(int(raw_data[2 + j]))
        skel_i.skeleton_point_indices = pt_idx
        node_skel.append(skel_i)
    aux_json.skeleton_points = node_skel

    json_file0 = open(meta_json_fname, 'w')
    json_file0.write(meta_json.toJSON())
    json_file0.close()

    json_file1 = open(aux_json_fname, 'w')
    json_file1.write(aux_json.toJSON())
    json_file1.close()

if __name__=='__main__':
    if len(sys.argv) < 4:
        print('Usage: %s <metadata txt file> <metadata json file> <auxiliary json file>'%sys.argv[0])
        exit(1)
    meta_txt_fname  = sys.argv[1]
    meta_json_fname = sys.argv[2]
    aux_json_fname  = sys.argv[3]
    metadata_txt_to_json(meta_txt_fname, meta_json_fname, aux_json_fname)