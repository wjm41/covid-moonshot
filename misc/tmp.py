import pandas as pd
import numpy as np
# from selfies import decoder
from rdkit.Chem.Descriptors import MolWt
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, Draw, AllChem
import sys

# name = sys.argv[1]
# ind = int(sys.argv[2])
#
# orig = pd.read_csv('candidates_smiles.txt', header=None, names=['smiles'])
# orig = orig['smiles'].values[ind-1]
# df = pd.read_csv('mmp_'+str(ind)+'.txt', delim_whitespace=True)
# #df = pd.read_csv('opt2.smi', header=None, names=['smiles'])
# # print(df.head())
# #
# # df_new = pd.DataFrame(index=df.index, columns=['smiles'])
# #
# # n = -1
# # m = -1
# # for i, row in df.iterrows():
# #     # if i%4 == 0:
# #     #     n+=1
# #     if i%5 == 0:
# #         n+=1
# #     if i%20 ==0:
# #         m+=1
# #     # df_new.iloc[5*(i%4) + n] = row
# #     # print(i, n, 5*(i%4)+n)
# #     df_new.iloc[i%5 + 5*m + 25*(n%4)] = row
# #     print(MolFromSmiles(decoder(row['smiles'])))
# # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
# #     print(df_new)
# # df_new.to_csv('opt2.smi', index=False, header=False)
# #mols = [MolFromSmiles(decoder(x)) for x in df['smiles'].values]
# df['WtDiff'] = [MolWt(MolFromSmiles(x)) - MolWt(MolFromSmiles(orig)) for x in df['SMILES'].values]
# print(len(df))
# df = df[df['WtDiff']<60]
# print(len(df))
# mols = [MolFromSmiles(x) for x in np.random.choice(df['SMILES'].values, 100, replace=False)]
# g_mols = []
# for mol in mols:
#     if mol!=None:
#         g_mols.append(mol)
# for mol in g_mols:
#     tmp = AllChem.Compute2DCoords(mol)
# img=Draw.MolsToGridImage(mols,molsPerRow=5,subImgSize=(600,600))
# img.save(name+'.pdf')
# smarts = '[N:1]([#6:2])([#1])[#1].[#6:3][Br]>>[N:1]([#6:2])[#6:3]'
# smi_1 = 'Nc1cncc(N)c1'
# smi_2 = 'Cc1ccncc1Br'
# rxn = AllChem.ReactionFromSmarts(smarts)
# print(rxn.RunReactants((Chem.AddHs(MolFromSmiles(smi_1)),Chem.AddHs(MolFromSmiles(smi_2)))))
smarts = Chem.MolFromSmarts('[O][C][N]')
df = pd.read_csv('new_activities/activity_data_new.csv')
for smi in df['SMILES'].values:
    # print(smi)
    mol = Chem.MolFromSmiles(smi)
    match=mol.HasSubstructMatch(smarts)
    if match:
        # print(smi)
        print(df['CID'][df['SMILES']==smi].values[0])
