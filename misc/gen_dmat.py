from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
from rdkit.Chem import Draw, MolFromSmiles, AllChem
from sklearn.model_selection import train_test_split
from rdkit import DataStructs

name = sys.argv[1]

df_inactive = pd.read_csv(name+'_src.csv')

inds = np.load(name+'_inds_s4.npy')
inactive_inds = inds[:,1]

inactive_smi = df_inactive.iloc[inactive_inds]
inactive_smi = inactive_smi.drop_duplicates('smiles')
inactive_smi = inactive_smi['smiles'].values

inactive_mols = [MolFromSmiles(x) for x in inactive_smi]

df_active = pd.read_csv(name+'_tgt.csv')

active_inds = inds[:,0]

active_smi = df_active.iloc[active_inds]
active_smi = active_smi.drop_duplicates('smiles')
active_smi = active_smi['smiles'].values

active_mols = [MolFromSmiles(x) for x in active_smi]

mols = inactive_mols + active_mols

fprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024) for mol in mols]

D_matrix = np.empty((len(fprints), len(fprints)))

for i in tqdm(range(len(fprints))):
    for j in range(i,len(fprints)):
        D_matrix[i][j] = DataStructs.FingerprintSimilarity(fprints[i], fprints[j])

np.save('dmat_sars_s4.npy', D_matrix)
