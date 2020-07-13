import pandas as pd
import numpy as np
from rdkit.Chem import Draw, Lipinski, Descriptors, Crippen, MolFromSmiles, AllChem
from rdkit import Chem
#
# # def line_to_smiles(line):
# #     return ''.join(line.strip('\n').split(' '))
#
# # Read HTS results
# df_hts = pd.read_csv('new_activities/MproHTSresults.csv')
# df_hts = df_hts[['SMILES','Protease Assay (Screening): Corrected activity (%)']]
# df_hts = df_hts.rename(columns={'Protease Assay (Screening): Corrected activity (%)': 'activity'})
# df_hts['activity'] = df_hts['activity'].where(df_hts['activity']>50,1)
# df_hts['activity'] = df_hts['activity'].where(df_hts['activity']<50,0).astype(int)
# # df_hts.to_csv('new_activities/HTSresults.csv', index=False)
#
# # # Apply lipinski filtering
# # def passes_lipinski(mol):
# #     if not Descriptors.MolWt(mol) < 500:
# #         return False
# #     if not Crippen.MolLogP(mol) <= 5:
# #         return False
# #     if not Lipinski.NumHDonors(mol) <= 5:
# #         return False
# #     if not Lipinski.NumHAcceptors(mol) <= 10:
# #         return False
# #     return True
# #
# # df = pd.read_csv('sars.csv')
# # df_lip = pd.read_csv('sars_lip.csv')
# # print('orig_length: {}, lip-filtered length: {}'.format(len(df),len(df_lip)))
#
# # df['mol'] = [Chem.MolFromSmiles(smi) for smi in df['smiles'].values]
# # df['lip'] = [passes_lipinski(m) for m in df['mol'].values]
# # df_lip = df[df['lip']]
# # df_lip = df_lip[['smiles','activity']]
# # df_lip.to_csv('sars_lip.csv', index=False)
# from rdkit.Chem import Draw, MolFromSmiles
#
# # df = pd.read_csv('assay_results.csv').rename(columns={'% Inhibition at 20 mM (N=1)':'inhib_1', '% Inhibition at 20 mM (N=2)':'inhib_2'})
# # df['inhib_1'] = df['inhib_1'].where(df['inhib_1']!='No Inhibition',0).astype('float')
# # df['inhib_2'] = df['inhib_2'].where(df['inhib_2']!='No Inhibition',0).astype('float')
# # df['inhib_mean'] = (df['inhib_1'] + df['inhib_2'])/2
# # df['inhib_std'] = np.sqrt((np.square(df['inhib_1'] - df['inhib_mean']) + np.square(df['inhib_2'] - df['inhib_mean']))/2)
# # df = df[['SMILES','shipment_ID','inhib_mean','inhib_std']]
# # df.to_csv('new_activities/assay_results_processed.csv', index=False)
#
# # Read activity data to construct multitask
# df = pd.read_csv('new_activities/activity_data_new.csv')
# df['mol'] = [MolFromSmiles(x) for x in df['SMILES']]
# #acry = MolFromSmiles('[CX3](=[OX1])[CX3]=[CX3]')
# acry = MolFromSmiles('O=C(C=C)N')
# #chloroace = MolFromSmiles('[CX3](=[OX1])[CX4][Cl]')
# chloroace = MolFromSmiles('ClCC(=O)N')
#
# df['acry'] = [x.HasSubstructMatch(acry) for x in df['mol']]
# df['chloroace'] = [x.HasSubstructMatch(chloroace) for x in df['mol']]
# #
# df_active = df[df['f_avg_pIC50'].notnull()]
# df_inactive = df[df['f_avg_pIC50'].isnull()]
# #
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     #print(df['chloroace'])
#     print('Number of actives: {}'.format(len(df_active)))
#     print('Number of inactives: {}'.format(len(df_inactive)))
#
#     print('Number of acry: {}'.format(len(df[df['acry']])))
#     print('Number of chloroace: {}'.format(len(df[df['chloroace']])))
#     print('Number of both: {}'.format(len(df[df['acry'] & df['chloroace']])))
# df_acry_actives = df_active[df_active['acry']]
# df_acry_inactives = df_inactive[df_inactive['acry']]
# df_chloro_actives = df_active[df_active['chloroace']]
# df_chloro_inactives = df_inactive[df_inactive['chloroace']]
# df_rest_actives = df_active[~df_active['acry'] & ~df_active['chloroace']]
# df_rest_inactives = df_inactive[~df_inactive['acry'] & ~df_inactive['chloroace']]
# print('Old Numbers:')
# print('Number of acry actives: {}'.format(len(df_acry_actives)))
# print('Number of acry inactives: {}'.format(len(df_acry_inactives)))
# print('Number of chloro actives: {}'.format(len(df_chloro_actives)))
# print('Number of chloro inactives: {}'.format(len(df_chloro_inactives)))
# print('Number of rest actives: {}'.format(len(df_rest_actives)))
# print('Number of rest inactives: {}'.format(len(df_rest_inactives)))
# print('Length of original dataset: {}, sum of split datasets: {}'.format(len(df), len(df_acry_actives) +
#                                                                          len(df_acry_inactives) + len(df_chloro_actives) +
#                                                                          len(df_chloro_inactives) + len(df_rest_actives) +
#                                                                          len(df_rest_inactives)))
# df['activity'] = df['f_avg_pIC50'].where(df['f_avg_pIC50'].isnull(),1)
# df['activity'] = df['activity'].where(df['activity'].notnull(),0).astype(int)
# df = df[~((df['activity']==0) & (df['f_inhibition_at_50_uM']>50))]
#
# df_active = df[df['activity']==1]
# df_inactive = df[df['activity']==0]
# df_acry_actives = df_active[df_active['acry']]
# df_acry_inactives = df_inactive[df_inactive['acry']]
# df_chloro_actives = df_active[df_active['chloroace']]
# df_chloro_inactives = df_inactive[df_inactive['chloroace']]
# df_rest_actives = df_active[~df_active['acry'] & ~df_active['chloroace']]
# df_rest_inactives = df_inactive[~df_inactive['acry'] & ~df_inactive['chloroace']]
#
# print('New Numbers:')
# print('Number of acry actives: {}'.format(len(df_acry_actives)))
# print('Number of acry inactives: {}'.format(len(df_acry_inactives)))
# print('Acry activity: {:.2f}%'.format(100*len(df_acry_actives)/(len(df_acry_inactives)+len(df_acry_actives))))
# print('Number of chloro actives: {}'.format(len(df_chloro_actives)))
# print('Number of chloro inactives: {}'.format(len(df_chloro_inactives)))
# print('chloro activity: {:.2f}%'.format(100*len(df_chloro_actives)/(len(df_chloro_inactives)+len(df_chloro_actives))))
# print('Number of rest actives: {}'.format(len(df_rest_actives)))
# print('Number of rest inactives: {}'.format(len(df_rest_inactives)))
# print('rest activity: {:.2f}%'.format(100*len(df_rest_actives)/(len(df_rest_inactives)+len(df_rest_actives))))
#
# print('Length of original dataset: {}, sum of split datasets: {}'.format(len(df), len(df_acry_actives) +
#                                                                          len(df_acry_inactives) + len(df_chloro_actives) +
#                                                                          len(df_chloro_inactives) + len(df_rest_actives) +
#                                                                          len(df_rest_inactives)))
#
# df['acry_class'] = df['activity'].where(df['acry'],np.nan).astype('Int64')
# df['acry_reg'] = df['f_avg_pIC50'].where(df['acry'],np.nan)
# df['chloro_class'] = df['activity'].where(df['chloroace'],np.nan).astype('Int64')
# df['chloro_reg'] = df['f_avg_pIC50'].where(df['chloroace'],np.nan)
# df['rest_class'] = df['activity'].where(~df['acry'] & ~df['chloroace'],np.nan).astype('Int64')
# df['rest_reg'] = df['f_avg_pIC50'].where(~df['acry'] & ~df['chloroace'],np.nan)
# # df = df[['SMILES', 'acry_class', 'chloro_class', 'rest_class', 'acry_reg', 'chloro_reg', 'rest_reg']]
# # df.to_csv('new_activities/covid_multitask_pIC50.smi', index=False)
# # df = df.merge(df_hts, how='outer', on='SMILES')
# df['activity'] = df['activity'].astype('Int64')
# # df.to_csv('new_activities/covid_multitask_HTS.smi', index=False)
#
# df_acry = df[df['acry']]
# # df_acry['activity'] = df_acry['f_avg_IC50'].where(df_acry['f_avg_IC50'].isnull(),1)
# # df_acry['activity'] = df_acry['activity'].where(df_acry['activity'].notnull(),0).astype(int)
# # df_acry = df_acry[['SMILES','activity']]
# df_acry = df_acry[['SMILES','CID','activity','f_avg_IC50']]
# df_acry.to_csv('new_activities/acry_activity.smi', index=False)
# # df_acry = df_acry.drop(['mol','acry','chloroace'], axis=1)
# # df_acry.to_csv('new_activities/acry_activity.csv', index=False)
#
# df_chloro = df[df['chloroace']]
# # img=Draw.MolsToGridImage(df_chloro['mol'].values,molsPerRow=4,subImgSize=(400,400))
# # img.save('chloro_grid.png')
# # df_chloro['activity'] = df_chloro['f_avg_IC50'].where(df_chloro['f_avg_IC50'].isnull(),1)
# # df_chloro['activity'] = df_chloro['activity'].where(df_chloro['activity'].notnull(),0).astype(int)
# # df_chloro = df_chloro[['SMILES','activity']]
# df_chloro = df_chloro[['SMILES','CID','activity','f_avg_IC50']]
# df_chloro.to_csv('new_activities/chloroace_activity.smi', index=False)
# # df_chloro = df_chloro.drop(['mol','acry','chloroace'], axis=1)
# # df_chloro.to_csv('new_activities/chloroace_activity.csv', index=False)
#
# df_rest = df[~df['acry'] & ~df['chloroace']]
# # df_rest['activity'] = df_rest['f_avg_IC50'].where(df_rest['f_avg_IC50'].isnull(),1)
# # df_rest['activity'] = df_rest['activity'].where(df_rest['activity'].notnull(),0).astype(int)
# df_rest = df_rest[['SMILES','CID','activity','f_avg_IC50']]
# # df_rest = df_rest[['SMILES','activity']]
# df_rest.to_csv('new_activities/rest_activity.smi', index=False)
# # df_rest= df_rest.drop(['mol','acry','chloroace'], axis=1)
# # df_rest.to_csv('new_activities/rest_activity.csv', index=False)

# df_acry_actives = df_acry_actives.drop(['mol','acry','chloroace'], axis=1)
# df_acry_inactives = df_acry_inactives.drop(['mol','acry','chloroace'], axis=1)
# df_chloro_actives = df_chloro_actives.drop(['mol','acry','chloroace'], axis=1)
# df_chloro_inactives = df_chloro_inactives.drop(['mol','acry','chloroace'], axis=1)
# df_rest_actives = df_rest_actives.drop(['mol','acry','chloroace'], axis=1)
# df_rest_inactives = df_rest_inactives.drop(['mol','acry','chloroace'], axis=1)
#
# df_acry_actives.to_csv('new_activities/acry_actives.csv', index=False)
# df_acry_inactives.to_csv('new_activities/acry_inactives.csv', index=False)
# df_chloro_actives.to_csv('new_activities/chloro_actives.csv', index=False)
# df_chloro_inactives.to_csv('new_activities/chloro_inactives.csv', index=False)
# df_rest_actives.to_csv('new_activities/rest_actives.csv', index=False)
# df_rest_inactives.to_csv('new_activities/rest_inactives.csv', index=False)

# df = pd.read_csv('new_activities/rest_activity.smi')
# df = df[df['activity']==1]
# smiles_actives = df['SMILES'].values


# duplicates = set(smiles_actives) & set(smiles_list)
# print(duplicates)
# smiles_list = np.random.choice(df['smiles'].values, size=20, replace=False)

df_scored = pd.read_csv('new_activities/noncovalent_lib_final.txt')
# smiles_list = df_scored['SMILES'].values
smiles_list = np.random.choice(df_scored['SMILES'].values, size=100, replace=False)
print(smiles_list)
mols = [MolFromSmiles(x) for x in smiles_list]
for mol in mols:
    tmp = AllChem.Compute2DCoords(mol)
img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(300, 300))
with open('new_activities/random_noncovalents.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % smi for smi in smiles_list)
img.save('random_noncovalent_lib.png')

# for i in range(1,6):
#     df_scored = pd.read_csv('new_activities/top_'+str(i)+'.csv')
#     smiles_list = df_scored['SMILES'].values

    # duplicates = set(smiles_actives) & set(smiles_list)
    # print(duplicates)
    # smiles_list = np.random.choice(df['smiles'].values, size=20, replace=False)
    # mols = [MolFromSmiles(x) for x in smiles_list]
    # for mol in mols:
    # tmp = AllChem.Compute2DCoords(mol)
    # img=Draw.MolsToGridImage(mols,molsPerRow=5,subImgSize=(300,300))
    # img.save('scored_noncovalents_'+str(i)+'.png')
#