import pandas as pd

pose_df = pd.read_csv('smiles_poses.smi', delim_whitespace=True, header=None, names=['SMILES','dockvalent_id','TITLE'])
pose_df = pose_df[['SMILES','dockvalent_id']]
docking_df = pd.read_csv('covalent-docking-overlap.csv')
docking_df = docking_df[['SMILES','TITLE','dockvalent_id']]
docking_df = docking_df.drop_duplicates(subset='dockvalent_id')
#docking_df = docking_df.drop_duplicates(subset='SMILES')
print('Number of unique dock ids in dock_df: {}'.format(len(docking_df)))

pose_df = pose_df.drop_duplicates(subset='SMILES')
pose_df = pose_df.drop_duplicates(subset='dockvalent_id')
print('Number of unique SMILES & dock ids in pose_df: {}'.format(len(pose_df)))

#print(pose_df['dockvalent_id'][~pose_df['dockvalent_id'].isin(docking_df['dockvalent_id'])])
merged_df = pose_df.merge(docking_df, how='inner', on='dockvalent_id').rename(columns={'SMILES_x':'SMILES'})

merged_df = merged_df[['SMILES','TITLE','dockvalent_id']]
merged_df.to_csv('unique_docked_smiles.csv', index=False)