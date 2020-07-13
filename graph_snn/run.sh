#! /bin/bash
#while read p; do
#DATA=$(echo "$p")
#python mpi_soap.py $DATA
#done <data/Bender/dataset_names.txt
#python mpnn_multitask_desc.py -n_epochs 600 -n 10
#python mpnn_pair_HTS.py -n_epochs 30 -n_trials 3 -savename HTS_bigbatch -test -batch_size 64
python mpnn_pair_multi.py -n_epochs 30 -n_trials 1 -savename $1
#python mpnn_pair_score.py -modelname all_pairs -i $1 -target $2 
#python mpnn_sars_desc.py -n_trials 1 -n_epochs 500 -savename sars -test
#python gen_descs.py
