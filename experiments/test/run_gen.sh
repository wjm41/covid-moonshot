#! /bin/bash

init_pop=../../data/covid_submissions.csv
targets=../../data/xyz/all_ligands.xyz
mut_rate=0.01
n_gens=100
target_size=32
size_std=4
r_cut=3.5
a_sigm=0.3
kernel=rematch

python ../../GA-soap.py -csv ${init_pop} -tgt ${targets} -mut_rate ${mut_rate} \
    -n_gens ${n_gens} -tgt_size=${target_size} -size_stdev ${size_std} \
    -rcut ${r_cut} -sigma ${a_sigm} -kernel ${kernel}
