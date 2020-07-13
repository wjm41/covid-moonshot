#! /bin/bash
#for i in {25760016..25760022};
#do
#qdel $i
#done
a=($(qstat -u wjm41 | tail -n +6 | grep 'R' | sed 's/ .*//g'))
for i in "${a[@]}"; do
    echo $i
    qdel $i
done
