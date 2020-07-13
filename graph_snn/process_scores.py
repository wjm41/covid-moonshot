import argparse


import pandas as pd
import numpy as np
from rdkit import Chem
import sys

parser = argparse.ArgumentParser()

parser.add_argument('-output', type=str,
                    help='name for directory containing saved model params and tensorboard logs')
parser.add_argument('-target', type=str, default='acry',
                    help='target series for scoring hits')
parser.add_argument('-input', type=str,
                    help='input file of smiles to score relative to the targets.')
args = parser.parse_args()

#df = pd.read_csv(args.input, header=0, names=['SMILES','avg_score_1','avg_score_2','avg_score_3','avg_score_4','avg_score_5','avg_score','std'])
df = pd.read_csv(args.input, header=0, names=['SMILES','avg_score_2','avg_score_3','avg_score_4','avg_score_5','avg_score','std'])
smiles_list = df['SMILES'].values
df_actives = pd.read_csv('data/'+args.target+'_activity.smi')
df_actives = df_actives[df_actives['activity']==1]
smiles_actives = df_actives['SMILES'].values

bool_list = [(smi in smiles_actives) for smi in smiles_list]
bool_list = [not x for x in bool_list]
df = df[bool_list]
print(df)
df = df.sort_values(by=['avg_score'], ascending=False)
df = df.iloc[:100]
df.to_csv(args.output, index=False)
