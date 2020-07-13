import pandas as pd
from tqdm import tqdm
from rdkit.Chem import MolFromSmiles, MolToSmiles
from crem.crem import mutate_mol

# df = pd.read_csv('new_activities/rest_activity.smi')
# df = df[df['activity']==1]
# print('Number of non-covalent actives: {}'.format(len(df)))
# smiles_list = df['SMILES'].values
# actives = [MolFromSmiles(smi) for smi in smiles_list]
#
# df_lib = pd.read_csv('new_activities/best_new_lib.smi')
# smiles_list = df_lib['SMILES'].values
# print('Size of non-covalent SMARTS library: {}'.format(len(df_lib)))
# lib_mols = [MolFromSmiles(smi) for smi in smiles_list]
#
# input_list = list(set(actives + lib_mols))
# prod_list = []
# for m in tqdm(input_list):
#     prod_list = prod_list + list(mutate_mol(m, db_name='replacements02_sc2.5.db'))
# print(len(prod_list))
# with open('new_activities/mutated_lib_new.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % mol for mol in prod_list)

# df = pd.read_csv('new_activities/mutated_lib_new.txt')
# print(df)
# smiles_list = list(set(df['SMILES'].values))
# smiles_list = list(set(MolToSmiles(MolFromSmiles(smi)) for smi in tqdm(smiles_list)))
# df = pd.DataFrame(smiles_list, columns=['SMILES'])
# df.to_csv('new_activities/mutated_lib_unique_new.txt', index=False)
df = pd.read_csv('new_activities/mutated_lib_unique_new.txt')
print(df)