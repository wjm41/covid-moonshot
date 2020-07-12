import pandas as pd
from rdkit.Chem import MolFromSmiles, MolToSmiles

def find_duplicate_submissions(mol_list):
    df = pd.read_csv('data/covid_submissions_03_31_2020.csv')
    submissions = df['SMILES'].values
    #ID = df['CID'].values
    can_smiles_list = [MolToSmiles(MolFromSmiles(x)) for x in mol_list]
    n_dup=0
    for i,mol in enumerate(can_smiles_list):
        if mol in submissions:
            print('{}th molecule already in submissions: {},{}'.format(i+1, mol, df[df['SMILES']==mol]['CID'].values[0]))
            n_dup+=1
    print('Number of duplicated molecules: {}'.format(n_dup))

if __name__ == '__main__':
    my_mols = ['CC(=O)N1CCN(S(=O)(=O)Cc2ccccc2)CC1','CC(=O)NCCc2ccc(CN1CCN(C(C)=O)CC1)cc2','CC(=O)NCCc2cccc(CN1CCN(C(C)=O)CC1)c2',
               'CC(=O)N2CCN(Cc1ccc(CCS(N)(=O)=O)cc1)CC2','CC(=O)N3CCN(C(c1ccccc1)N2CCC(O)CC2)CC3','O=C(CN1CCNCC1)Nc2ccccc2','CC(=O)N2CCN(CCCc1ccc(S(N)(=O)=O)cc1)CC2'
               ,'CC(=O)NCc1cc(C#N)ccc1CNC(=O)N2CCOCC2','CC(=O)NCc1cscc1C2CCCC2C(N)=O','CC(=O)NC(c1cccc(Cl)c1)c2nnc(C)s2','CC(=O)NC(c1cccc(Cl)c1)N2CCOCC2',
                'CC(=O)NCCc1c[nH]c2c(CN(C)C(C)=O)cc(F)cc12','CC(=O)NC(CNS(C)(=O)=O)c1cccc(Cl)c1']

    find_duplicate_submissions(my_mols)
    #find_duplicate_submissions(['Cc1ccc(OCC(=O)N2CCN(CCCCC(=O)Nc3cnccc3C)CC2)cc1'])