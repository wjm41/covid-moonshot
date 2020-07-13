import pandas as pd
from pubchempy import Compound

df = pd.read_csv('AID_1706_datatable_all.csv')
print(df.head())
df['SMILES'] = [Compound.from_cid(cid).isomeric_smiles for cid in df['PUBCHEM_CID']]
df.to_csv('AID_1706_datatable_all.csv', index=False)
print(df.head())