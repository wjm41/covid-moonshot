import argparse


import pandas as pd
import numpy as np
from rdkit import Chem
import sys

#parser = argparse.ArgumentParser()

#parser.add_argument('-output', type=str,
#                    help='name for directory containing saved model params and tensorboard logs')
#parser.add_argument('-target', type=str, default='acry',
#                    help='target series for scoring hits')
#parser.add_argument('-input', type=str,
#                    help='input file of smiles to score relative to the targets.')
#args = parser.parse_args()

taskname = sys.argv[1]
#df1 = pd.read_csv(taskname+'_model_1_scores.csv').rename(columns={'avg_score': 'avg_score_1'})
df2 = pd.read_csv(taskname+'_model_2_scores.csv').rename(columns={'avg_score': 'avg_score_2'})
df3 = pd.read_csv(taskname+'_model_3_scores.csv').rename(columns={'avg_score': 'avg_score_3'})
df4 = pd.read_csv(taskname+'_model_4_scores.csv').rename(columns={'avg_score': 'avg_score_4'})
df5 = pd.read_csv(taskname+'_model_5_scores.csv').rename(columns={'avg_score': 'avg_score_5'})
#df4 = pd.read_csv(taskname+'_model_4_scores.csv').rename(columns={'avg_score': 'score_4'})
#df5 = pd.read_csv(taskname+'_model_5_scores.csv').rename(columns={'avg_score': 'score_5'})

#print(pd.concat([df1, df2, df3, df4, df5], axis=1)) 
#df1 = df1.merge(df2, on='SMILES')
#df1 = df1.merge(df3, on='SMILES')
#df1 = df1.merge(df4, on='SMILES')
#df1 = df1.merge(df5, on='SMILES')
##df1['avg_score'] = df1[['score_1','score_2','score_3','score_4','score_5']].mean(axis=1)
##df1['std'] = df1[['score_1','score_2','score_3','score_4','score_5']].std(axis=1)
#df1['ensemble_avg_score'] = df1[['avg_score_1','avg_score_2','avg_score_3', 'avg_score_4','avg_score_5']].mean(axis=1)
#df1['ensemble_std'] = df1[['avg_score_1','avg_score_2','avg_score_3','avg_score_4','avg_score_5']].std(axis=1)
##df1 = df1.rename(columns={'avg_score_x': 'score_1'})
#print(df1)
#df1 = df1[['SMILES','avg_score_1','avg_score_2','avg_score_3','avg_score_4','avg_score_5','ensemble_avg_score','ensemble_std']]
##df1.to_csv('expanded_acrylib_ensemble.csv', index=False)
#df1.to_csv('expanded_noncovalent_ensemble.csv', index=False)
df2 = df2.merge(df3, on='SMILES')
df2 = df2.merge(df4, on='SMILES')
df2 = df2.merge(df5, on='SMILES')
df2['ensemble_top_score'] = df2[['avg_score_2','avg_score_3', 'avg_score_4','avg_score_5']].mean(axis=1)
df2['ensemble_std'] = df2[['avg_score_2','avg_score_3','avg_score_4','avg_score_5']].std(axis=1)
print(df2)
df2 = df2[['SMILES','avg_score_2','avg_score_3','avg_score_4','avg_score_5','ensemble_top_score','ensemble_std']]
df2.to_csv('expanded_noncovalent_ensemble4.csv', index=False)
