#!/usr/bin/bash

./example_H2.exe
./metadata_txt_to_json.py Coulomb_3D_1e-6.txt Coulomb_3D_1e-6_meta.json Coulomb_3D_1e-6_aux.json
mv Coulomb_3D_1e-6.txt Coulomb_3D_1e-6-bak.txt
./metadata_json_to_txt.py Coulomb_3D_1e-6.txt Coulomb_3D_1e-6_meta.json Coulomb_3D_1e-6_aux.json
./example_read_H2_file.exe