#! /bin/bash

init_pop=../../data/covid_submissions.csv
target1=../../data/xyz/site11_ligands.xyz
target2=../../data/xyz/site2_ligands.xyz
mut_rate=0.015
n_gens=50
target_size=30
size_std=4
r_cut=3.5
a_sigm=0.3
kernel=rematch

python ../../GA-soap.py -csv ${init_pop} -tgt ${target1} -tgt2 ${target2} \
    -mut_rate ${mut_rate} -n_gens ${n_gens} -tgt_size=${target_size} \
    -size_stdev ${size_std} -rcut ${r_cut} -sigma ${a_sigm} -kernel ${kernel}
