#! /bin/bash
#for j in {1..5};
#do
j=$1
for i in {0..19}; 
do
qsub slurm_amide $j $i
done
#done
