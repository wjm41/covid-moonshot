import pandas as pd
import numpy as np
from rdkit import DataStructs
from rdkit.Chem import Draw, MolFromSmiles, AllChem
from tqdm import tqdm
from hurry.filesize import size

import matplotlib.pyplot as plt

#df = pd.read_csv('sars.csv')
df = pd.read_csv('data/HTSresults.csv')
for x in df['SMILES']:
    print(x)
    print(MolFromSmiles(x))
df['mol'] = df['SMILES'].apply(lambda x: MolFromSmiles(x))

df_active = df[df['activity']==1]
df_inactive = df[df['activity']==0]
#
# df_active['mol'] = df_active['smiles'].apply(lambda x: MolFromSmiles(x))
# df_active = df_active[df['mol'].notnull()]

fprints_active = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024) for mol in df_active['mol'].values]
fprints_inactive = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024) for mol in df_inactive['mol'].values]
#
D_matrix = np.empty((len(fprints_active), len(fprints_inactive)))
print('D_matrix size: {}'.format(size(D_matrix.nbytes)))
for i in tqdm(range(len(fprints_active))):
    for j in range(len(fprints_inactive)):
        D_matrix[i][j] = DataStructs.FingerprintSimilarity(fprints_active[i], fprints_inactive[j])

np.save('/rds-d2/user/wjm41/hpc-work/HTS_D_mat.npy', D_matrix)

for row in D_matrix:
    print(np.argmax(row))
# print(D_matrix)
# plt.clf()
# plt.figure(figsize=(7, 7))
# plt.matshow(D_matrix, fignum=1)
# cb = plt.colorbar(fraction=0.046, pad=0.04)
# plt.show()

# df['mol'] = df['SMILES'].apply(lambda x: MolFromSmiles(x))
#df = df[df['mol'].notnull()]

# fprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024) for mol in df['mol'].values]

# rows = []
# cols = []
# threshold = 0.4
# for i in tqdm(range(len(fprints))):
#     for j in range(i, len(fprints)):
#         sim = DataStructs.FingerprintSimilarity(fprints[i], fprints[j])
#         if sim>threshold:
#             rows.append(i)
#             cols.append(j)
# rows = np.array(rows)
# cols = np.array(cols)
# rows = rows.reshape(-1,1)
# cols = cols.reshape(-1,1)
# inds = np.hstack([rows, cols])
#
# np.save('similar_inds.npy', inds)
#
# df_rows = df.iloc[rows.flatten()]['pIC50'].values
# df_cols = df.iloc[cols.flatten()]['pIC50'].values
#
# diff = np.absolute(df_rows - df_cols)
# print(diff)
#
# diff = np.where(diff>0.2, 1, 0)
# num_pairs = np.sum(diff)
# print('Number of increased activity similar-pairs: {}'.format(num_pairs))
