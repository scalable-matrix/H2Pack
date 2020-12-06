#!/usr/bin/env python3

import json
import sys

class EmptyClass(object):
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=2)

def metadata_txt_to_json(txt_file_name, json_file_name):
    txt_file = open(txt_file_name, 'r')
    lines = txt_file.readlines()
    txt_file.close()

    json_data = EmptyClass()
    json_data.dim_point                 = int(lines[0])
    json_data.dim_kernel                = int(lines[1])
    json_data.num_point                 = int(lines[2])
    json_data.size_matrix               = int(lines[3])
    json_data.is_symmetric              = int(lines[4])
    json_data.max_points_in_leaf_node   = int(lines[5])
    json_data.num_node                  = int(lines[6])
    json_data.root_node_index           = int(lines[7])
    json_data.num_level                 = int(lines[8])
    json_data.is_HSS_matrix             = int(lines[9])
    json_data.min_admissible_pair_level = int(lines[10])
    json_data.num_inadmissible_pair     = int(lines[11])
    json_data.num_admissible_pair       = int(lines[12])
    json_data.has_partially_adm_pair    = int(lines[13])
    json_data.relative_accuracy         = float(lines[14])
    curr_row = 15

    nodes = []
    num_leaf_node = 0
    for i in range(json_data.num_node):
        raw_data = [x for x in lines[curr_row + i].split(' ') if x]
        node_i = EmptyClass()
        node_i.index = int(raw_data[0])
        node_i.level = int(raw_data[1])
        node_i.point_cluster_first_index = int(raw_data[2])
        node_i.point_cluster_last_index  = int(raw_data[3])
        node_i.num_children = int(raw_data[4])
        if 0 == node_i.num_children:
            num_leaf_node += 1
        node_i.children = []
        for j in range(node_i.num_children):
            node_i.children.append(int(raw_data[5 + j]))
        nodes.append(node_i)
    json_data.nodes = nodes
    curr_row += json_data.num_node

    U_mat = []
    for i in range(json_data.num_node):
        raw_data = [x for x in lines[curr_row + i].split(' ') if x]
        U_i = EmptyClass()
        U_i.node    = int(raw_data[0])
        U_i.num_row = int(raw_data[1])
        U_i.num_col = int(raw_data[2])
        U_mat.append(U_i)
    json_data.U_matrices = U_mat
    curr_row += json_data.num_node

    B_mat = []
    for i in range(json_data.num_admissible_pair):
        raw_data = [x for x in lines[curr_row + i].split(' ') if x]
        B_i = EmptyClass()
        B_i.node_row    = int(raw_data[0])
        B_i.node_col    = int(raw_data[1])
        B_i.num_row     = int(raw_data[2])
        B_i.num_col     = int(raw_data[3])
        B_i.is_part_adm = int(raw_data[4])
        B_mat.append(B_i)
    json_data.B_matrices = B_mat
    curr_row += json_data.num_admissible_pair

    D_mat = []
    for i in range(num_leaf_node + json_data.num_inadmissible_pair):
        raw_data = [x for x in lines[curr_row + i].split(' ') if x]
        D_i = EmptyClass()
        D_i.node_row = int(raw_data[0])
        D_i.node_col = int(raw_data[1])
        D_i.num_row  = int(raw_data[2])
        D_i.num_col  = int(raw_data[3])
        D_mat.append(D_i)
    json_data.D_matrices = D_mat
    curr_row += (num_leaf_node + json_data.num_inadmissible_pair)

    json_data.has_extended_fields = int(lines[curr_row])
    if 1 == json_data.has_extended_fields:
        node_skel = []
        curr_row += 1
        for i in range(json_data.num_node):
            raw_data = [x for x in lines[curr_row + i].split(' ') if x]
            skel_i = EmptyClass()
            skel_i.node = int(raw_data[0])
            skel_i.num_skeleton_point = int(raw_data[1])
            pt_idx = []
            for j in range(skel_i.num_skeleton_point):
                pt_idx.append(int(raw_data[2 + j]))
            skel_i.skeleton_point_indices = pt_idx
            node_skel.append(skel_i)
        json_data.skeleton_points = node_skel

    json_file = open(json_file_name, 'w')
    json_file.write(json_data.toJSON())
    json_file.close()

if __name__=='__main__':
    if len(sys.argv) < 3:
        print('Usage: %s <txt metadata file> <json metadata file>'%sys.argv[0])
        exit(1)
    metadata_txt_to_json(sys.argv[1], sys.argv[2])