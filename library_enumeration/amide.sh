#! /bin/bash
for i in {0..9}; 
do
#echo $i
qsub slurm_soap $i
done
