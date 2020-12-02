#!/usr/bin/env python3

import json
import sys

def metadata_json_to_txt(json_file_name, txt_file_name):
    json_file = open(json_file_name, 'r')
    json_data = json.load(json_file)
    json_file.close()

    with open(txt_file_name, 'w') as f:
        f.write('%d\n'%json_data['point_dimension'])
        f.write('%d\n'%json_data['kernel_dimension'])
        f.write('%d\n'%json_data['num_point'])
        f.write('%d\n'%json_data['kernel_matrix_size'])
        f.write('%d\n'%json_data['symmetric_kernel_matrix'])
        f.write('%d\n'%json_data['max_points_in_leaf_node'])
        f.write('%d\n'%json_data['num_node'])
        f.write('%d\n'%json_data['root_node_index'])
        f.write('%d\n'%json_data['num_level'])
        f.write('%d\n'%json_data['is_HSS_matrix'])
        f.write('%d\n'%json_data['min_admissible_pair_level'])
        f.write('%d\n'%json_data['num_inadmissible_pair'])
        f.write('%d\n'%json_data['num_admissible_pair'])
        f.write('%e\n'%json_data['relative_accuracy'])

        max_children = 2 ** json_data['point_dimension']
        nodes = json_data['nodes']
        num_leaf_node = 0
        for i in range(json_data['num_node']):
            node_i = nodes[i]
            f.write('%6d '%node_i['index'])
            f.write('%2d '%node_i['level'])
            f.write('%8d '%node_i['point_cluster_first_index'])
            f.write('%8d '%node_i['point_cluster_last_index'])
            f.write('%2d '%node_i['num_children'])
            node_i_children = node_i['children']
            node_i_n_children = node_i['num_children']
            if 0 == node_i_n_children:
                num_leaf_node += 1
            for j in range(node_i_n_children):
                f.write('%8d '%node_i_children[j])
            for j in range(node_i_n_children, max_children):
                f.write('-1 ')
            f.write('\n')

        U_mat = json_data['U_matrices']
        for i in range(json_data['num_node']):
            U_i = U_mat[i]
            f.write('%6d '%U_i['node'])
            f.write('%5d '%U_i['num_row'])
            f.write('%5d\n'%U_i['num_column'])

        B_mat = json_data['B_matrices']
        for i in range(json_data['num_admissible_pair']):
            B_i = B_mat[i]
            f.write('%6d '%B_i['row_node'])
            f.write('%6d '%B_i['column_node'])
            f.write('%5d '%B_i['num_row'])
            f.write('%5d\n'%B_i['num_column'])

        D_mat = json_data['D_matrices']
        for i in range(num_leaf_node + json_data['num_inadmissible_pair']):
            D_i = D_mat[i]
            f.write('%6d '%D_i['row_node'])
            f.write('%6d '%D_i['column_node'])
            f.write('%5d '%D_i['num_row'])
            f.write('%5d\n'%D_i['num_column'])

        f.write('%d\n'%json_data['has_extended_fields'])
        if 1 == json_data['has_extended_fields']:
            node_skel = json_data['node_skeleton_points']
            for i in range(json_data['num_node']):
                node_i_skel = node_skel[i]
                f.write('%6d '%node_i_skel['node'])
                f.write('%6d '%node_i_skel['num_skeleton_point'])
                skel_idx = node_i_skel['skeleton_point_indices']
                for j in range(node_i_skel['num_skeleton_point']):
                    f.write('%d '%skel_idx[j])
                f.write('\n')


if __name__=='__main__':
    if len(sys.argv) < 3:
        print('Usage: %s <json metadata file> <txt metadata file>'%sys.argv[0])
        exit(1)
    metadata_json_to_txt(sys.argv[1], sys.argv[2])