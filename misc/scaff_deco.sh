#! /bin/bash
python slice_db.py -i $1/$1.smi -u $1/$1_sliced.smi -s hr -f conditions.json.example 
python create_randomized_smiles.py -i $1/$1_sliced.smi -o $1/training -n 20 -d multi
python create_model.py -i $1/training/001.smi -o /rds-d2/user/wjm41/hpc-work/models/scaffold_decorator/$1/models/model.empty -d 0.2
python train_model.py -i /rds-d2/user/wjm41/hpc-work/models/scaffold_decorator/$1/models/model.empty -o /rds-d2/user/wjm41/hpc-work/models/scaffold_decorator/$1/models/model.trained -s $1/training/ -e 100 -b 64 --sen 10
python sample_scaffolds.py -m /rds-d2/user/wjm41/hpc-work/models/scaffold_decorator/$1/models/model.trained.90 -i $1/$1_scaffolds.smi -o $1_generated.csv -d multi -n $2 -r $3
