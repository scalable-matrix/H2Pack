#!/usr/bin/env python3

import json
import sys
import struct

class EmptyClass(object):
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=2)

def double_to_hex(f):
    return hex(struct.unpack('<Q', struct.pack('<d', f))[0])

def metadata_json_to_txt(meta_txt_fname, meta_json_fname, aux_json_fname):
    meta_json_file = open(meta_json_fname, 'r')
    meta_json = json.load(meta_json_file)
    meta_json_file.close()

    if aux_json_fname:
        aux_json_file = open(aux_json_fname, 'r')
        aux_json_tmp = json.load(aux_json_file)
        aux_json_file.close()
    else:
        print('No auxiliary JSON file, will use fake values for corresponding key/value pairs\n')
        aux_json_tmp_class = EmptyClass()
        aux_json_tmp = aux_json_tmp_class.toJSON()

    # If a field is missing in aux_json_tmp, use a fake value
    aux_json = EmptyClass()
    # C.1 dim_point
    if 'dim_point' in aux_json_tmp:
        aux_json.dim_point = aux_json_tmp['dim_point']
    else:
        aux_json.dim_point = 3
    # C.2 dim_kernel
    if 'dim_kernel' in aux_json_tmp:
        aux_json.dim_kernel = aux_json_tmp['dim_kernel']
    else:
        aux_json.dim_kernel = 1
    # C.3 num_point
    if 'num_point' in aux_json_tmp:
        aux_json.num_point = aux_json_tmp['num_point']
    else:
        aux_json.num_point = meta_json['nrow_matrix']
    # C.4 is_HSS
    if 'is_HSS' in aux_json_tmp:
        aux_json.is_HSS = aux_json_tmp['is_HSS']
    else:
        aux_json.is_HSS = 0
    # C.5 min_adm_level
    if 'min_adm_level' in aux_json_tmp:
        aux_json.min_adm_level = aux_json_tmp['min_adm_level']
    else:
        aux_json.min_adm_level = 2
    # C.6 max_leaf_points
    if 'max_leaf_points' in aux_json_tmp:
        aux_json.max_leaf_points = aux_json_tmp['max_leaf_points']
    else:
        aux_json.max_leaf_points = 300
    # C.7 QR_stop_tol
    if 'QR_stop_tol' in aux_json_tmp:
        aux_json.QR_stop_tol = aux_json_tmp['QR_stop_tol']
    else:
        aux_json.QR_stop_tol = 0
    # C.8 has_skeleton_points
    if 'has_skeleton_points' in aux_json_tmp:
        aux_json.has_skeleton_points = aux_json_tmp['has_skeleton_points']
    else:
        aux_json.has_skeleton_points = 0
    # C.9 point_coordinate
    if 'point_coordinate' in aux_json_tmp:
        aux_json.point_coordinate = aux_json_tmp['point_coordinate']
    else:
        aux_json.point_coordinate = [0.0 for i in range(aux_json.dim_point * aux_json.num_point)]
    # C.10 permutation_array
    if 'permutation_array' in aux_json_tmp:
        aux_json.permutation_array = aux_json_tmp['permutation_array']
    else:
        aux_json.permutation_array = [i for i in range(aux_json.num_point)]
    # C.11 skeleton_points
    if 'skeleton_points' in aux_json_tmp:
        aux_json.skeleton_points = aux_json_tmp['skeleton_points']
    else:
        aux_json.skeleton_points = []

    with open(meta_txt_fname, 'w') as f:
        # 1. Metadata: H2 / HSS common part
        f.write('%d\n'%aux_json.dim_point)          # C.1 dim_point
        f.write('%d\n'%aux_json.dim_kernel)         # C.2 dim_kernel
        f.write('%d\n'%aux_json.num_point)          # C.3 num_point
        f.write('%d\n'%meta_json['nrow_matrix'])    # A.1 nrow_matrix
        f.write('%d\n'%meta_json['ncol_matrix'])    # A.2 ncol_matrix
        f.write('%d\n'%meta_json['is_symmetric'])   # A.3 is_symmetric
        f.write('%d\n'%meta_json['num_node_row'])   # A.4 num_node_row
        f.write('%d\n'%meta_json['num_node_col'])   # A.5 num_node_col
        f.write('%d\n'%meta_json['root_node_row'])  # A.6 root_node_row
        f.write('%d\n'%meta_json['root_node_col'])  # A.7 root_node_col
        f.write('%d\n'%meta_json['num_level_row'])  # A.8 num_level_row
        f.write('%d\n'%meta_json['num_level_col'])  # A.9 num_level_col           
        f.write('%d\n'%aux_json.is_HSS)             # C.4 is_HSS
        f.write('%d\n'%aux_json.min_adm_level)      # C.5 min_adm_level
        num_leaf_node = 0
        nodes_row = meta_json['nodes_row']
        for i in range(meta_json['num_node_row']):
            node_i = nodes_row[i]
            node_i_n_children = node_i['num_children']
            if 0 == node_i_n_children:
                num_leaf_node += 1
        num_inadm_blks = meta_json['num_inadmissible_blocks'] - num_leaf_node
        f.write('%d\n'%num_inadm_blks)                      # A.14 num_inadmissible_blocks - num_leaf_node
        f.write('%d\n'%meta_json['num_admissible_blocks'])  # A.15 num_admissible_blocks
        f.write('%d\n'%meta_json['has_partial_adm_blocks']) # A.16 has_partial_adm_blocks

        # 2. Metadata: partitioning tree
        # A.10 nodes_row; A.11 nodes_col == NULL since H2 matrix is symmetric
        max_children = 2 ** aux_json.dim_point
        nodes_row = meta_json['nodes_row']
        for i in range(meta_json['num_node_row']):
            node_i = nodes_row[i]
            f.write('%6d '%node_i['index'])         # A.10.1 index
            f.write('%2d '%node_i['level'])         # A.10.2 level
            f.write('%8d '%node_i['cluster_head'])  # A.10.3 cluster_head
            f.write('%8d '%node_i['cluster_tail'])  # A.10.4 cluster_tail
            f.write('%2d '%node_i['num_children'])  # A.10.5 num_children
            node_i_children   = node_i['children']
            node_i_n_children = node_i['num_children']
            # A.10.6 children
            for j in range(node_i_n_children):
                f.write('%8d '%node_i_children[j])
            for j in range(node_i_n_children, max_children):
                f.write('-1 ')
            f.write('\n')

        # 3. Metadata data: U matrices
        # A.12 basis_matrices_row (A.13 ignored since H2 matrix is symmetric)
        U_mat = meta_json['basis_matrices_row']
        for i in range(meta_json['num_node_row']):
            U_i = U_mat[i]
            f.write('%6d '%U_i['node'])     # A.12.1 node
            f.write('%5d '%U_i['num_row'])  # A.12.2 num_row
            f.write('%5d\n'%U_i['num_col']) # A.12.3 num_col

        # 4. Metadata data: B matrices
        B_mat = meta_json['B_matrices']
        for i in range(meta_json['num_admissible_blocks']):
            B_i = B_mat[i]
            f.write('%6d '%B_i['node_row'])     # A.17.1 node_row
            f.write('%6d '%B_i['node_col'])     # A.17.2 node_col
            f.write('%5d '%B_i['num_row'])      # A.17.3 num_row
            f.write('%5d '%B_i['num_col'])      # A.17.4 num_col
            f.write('%d\n'%B_i['is_part_adm'])  # A.17.5 is_part_adm

        # 5. Metadata data: D matrices
        D_mat = meta_json['D_matrices']
        for i in range(meta_json['num_inadmissible_blocks']):
            D_i = D_mat[i]
            f.write('%6d '%D_i['node_row']) # A.18.1 node_row
            f.write('%6d '%D_i['node_col']) # A.18.2 node_col
            f.write('%5d '%D_i['num_row'])  # A.18.3 num_row
            f.write('%5d\n'%D_i['num_col']) # A.18.4 num_col

        # 6. Other necessary information for H2Pack
        f.write('%d\n'%aux_json.max_leaf_points)        # C.6 max_leaf_points
        f.write('%e\n'%aux_json.QR_stop_tol)            # C.7 QR_stop_tol
        f.write('%d\n'%aux_json.has_skeleton_points)    # C.8 has_skeleton_points
        # C.9 point_coordinate
        # Cast it to uint64_t so reading it from text file in MATLAB won't be so slow
        for i in range(aux_json.num_point):
            for j in range(aux_json.dim_point):
                cij = double_to_hex(aux_json.point_coordinate[i * aux_json.dim_point + j])
                f.write('%s '%cij[2:])
            f.write('\n')
        # C.10 permutation_array
        for i in range(aux_json.num_point):
            f.write('%d\n'%aux_json.permutation_array[i])
        # C.11 skeleton_point
        if 1 == aux_json.has_skeleton_points:
            node_skel = aux_json.skeleton_points
            for i in range(meta_json['num_node_row']):
                node_i_skel = node_skel[i]
                f.write('%6d '%node_i_skel['node'])
                f.write('%6d '%node_i_skel['num_skeleton_point'])
                skel_idx = node_i_skel['skeleton_point_indices']
                for j in range(node_i_skel['num_skeleton_point']):
                    f.write('%d '%skel_idx[j])
                f.write('\n')


if __name__=='__main__':
    if len(sys.argv) < 3:
        print('Usage: %s <metadata txt file> <metadata json file> <auxiliary json file>\n'%sys.argv[0])
        print('       <auxiliary json file> could be empty if it does not exist\n')
        exit(1)
    meta_txt_fname  = sys.argv[1]
    meta_json_fname = sys.argv[2]
    if len(sys.argv) == 4:
        aux_json_fname = sys.argv[3]
    else:
        aux_json_fname  = ""
    metadata_json_to_txt(meta_txt_fname, meta_json_fname, aux_json_fname)