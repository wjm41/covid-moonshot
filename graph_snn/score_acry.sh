#! /bin/bash
for j in {1..5};
do
for i in {0..9}; 
do
qsub slurm_acry $j $i
done
done
