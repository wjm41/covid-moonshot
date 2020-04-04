import argparse

import pandas as pd
from rdkit.Chem import Draw, MolFromSmiles
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmilesFromSmiles

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):
    data = pd.read_csv('data/unique_docked_smiles.csv')
    # filter out invalid SMILES
    data['mol'] = data['SMILES'].apply(lambda x: MolFromSmiles(x))
    data = data[data['mol'].notnull()]
    print('Number of unique SMILES: {}'.format(len(data)))
    # Create dictionary of key: scaffold and value: list of smiles
    scaffold_dict = {}
    scaff_list = []
    for i, smiles in enumerate(data['SMILES'].values):
        scaffold = MurckoScaffoldSmilesFromSmiles(smiles)
        scaff_list.append(scaffold)
        if scaffold not in scaffold_dict.keys():
            scaffold_dict[scaffold] = [smiles]
        else:
            smiles_list = scaffold_dict[scaffold]
            scaffold_dict[scaffold].append(smiles)
        #print(scaffold_dict)
    data['scaffold'] = scaff_list
    data[['SMILES','TITLE','dockvalent_id','scaffold']].to_csv('data/unique_docked_smiles_by_scaffold.csv', index=False)

    scaff_list = []
    print('\nNumber of distinct scaffolds: {}'.format(len(scaffold_dict)))
    n_scaff = 0
    for scaffold in sorted(scaffold_dict, key=lambda k: len(scaffold_dict[k]), reverse=True):
        n_scaff += 1
        if n_scaff<=20:
            print('#{} scaffold: {}, size: {}'.format(n_scaff, scaffold, len(scaffold_dict[scaffold])))
            scaff_list.append(scaffold)
        else:
            break
    scaff_mols = [MolFromSmiles(scaffold) for scaffold in scaff_list]
    img = Draw.MolsToGridImage(scaff_mols, molsPerRow=5, subImgSize=(200,200),
                               legends=['counts = {}'.format(len(scaffold_dict[scaffold])) for scaffold in scaff_list])
    img.save('data/scaff_grid.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_mat', type=str2bool, default=True,
                        help='True/False for loading Tanimoto distance matrix instead of calculating from scratch')
    parser.add_argument('-plot', type=str2bool, default=False,
                        help='True/False for plotting Tanimoto distance and similarities')
    parser.add_argument('-K', type=int, default=20,
                        help='number of clusters in k-medoids algorithm')
    args = parser.parse_args()

    main(args)