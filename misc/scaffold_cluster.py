import argparse

import pandas as pd
from rdkit.Chem import Draw, MolFromSmiles, MolToSmiles
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmilesFromSmiles, MakeScaffoldGeneric


def main(args):
    data = pd.read_csv('data/unique_docked_smiles.csv')
    # filter out invalid SMILES
    data['mol'] = data['SMILES'].apply(lambda x: MolFromSmiles(x))
    data = data[data['mol'].notnull()]
    print('Number of unique SMILES: {}'.format(len(data)))
    # Create dictionary of key: scaffold and value: list of smiles
    scaffold_dict = {}
    scaff_list = []
    scaffold_generic_dict = {}
    scaff_generic_list = []
    for i, smiles in enumerate(data['SMILES'].values):
        scaffold = MurckoScaffoldSmilesFromSmiles(smiles)
        scaffold_generic = MolToSmiles(MakeScaffoldGeneric(MolFromSmiles(scaffold)))
        #print(scaffold)
        #print(scaffold_generic)
        scaff_list.append(scaffold)
        scaff_generic_list.append(scaffold_generic)
        if scaffold not in scaffold_dict.keys():
            scaffold_dict[scaffold] = [smiles]
        else:
            scaffold_dict[scaffold].append(smiles)
        if scaffold_generic not in scaffold_generic_dict.keys():
            scaffold_generic_dict[scaffold_generic] = [smiles]
        else:
            scaffold_generic_dict[scaffold_generic].append(smiles)
        #print(scaffold_dict)
    data['scaffold'] = scaff_list
    data['scaffold_generic'] = scaff_generic_list
    data.drop(columns='mol').to_csv('data/unique_docked_smiles_by_scaffold.csv', index=False)

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

    scaff_generic_list = []
    print('\nNumber of distinct generic scaffolds: {}'.format(len(scaffold_generic_dict)))
    n_scaff = 0
    for scaffold_gen in sorted(scaffold_generic_dict, key=lambda k: len(scaffold_generic_dict[k]), reverse=True):
        n_scaff += 1
        if n_scaff<=20:
            print('#{} generic scaffold: {}, size: {}'.format(n_scaff, scaffold_gen, len(scaffold_generic_dict[scaffold_gen])))
            scaff_generic_list.append(scaffold_gen)
        else:
            break
    scaffgen_mols = [MolFromSmiles(scaffold) for scaffold in scaff_generic_list]
    img = Draw.MolsToGridImage(scaffgen_mols, molsPerRow=5, subImgSize=(200,200),
                               legends=['counts = {}'.format(len(scaffold_generic_dict[scaffold])) for scaffold in scaff_generic_list])
    img.save('data/scaff_generic_grid.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_mat', action='store_true',
                        help='True/False for loading Tanimoto distance matrix instead of calculating from scratch')
    parser.add_argument('-plot', action='store_true',
                        help='True/False for plotting Tanimoto distance and similarities')
    parser.add_argument('-K', type=int, default=20,
                        help='number of clusters in k-medoids algorithm')
    args = parser.parse_args()

    main(args)
