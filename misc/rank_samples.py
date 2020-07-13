import argparse

import pandas as pd
import numpy as np
from rdkit import DataStructs
from rdkit.Chem import MolFromSmiles, AllChem
import matplotlib.pyplot as plt


def main(args):
    dataset = pd.read_csv(args.csv, usecols=['SMILES', 'CID', 'source', 'dock_score'])
    #enamine_df = pd.read_csv('data/processed_data_enamine.csv', usecols=['SMILES', 'CID', 'source', 'dock_score'])

    #dataset = pd.merge(enamine_df, dataset, how='outer')

    mols = [MolFromSmiles(smiles) for smiles in dataset['SMILES'].values]
    fprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024) for mol in mols]

    mol_set = [ [fprints[i], row['dock_score'], row['SMILES'], row['CID'], row['source']] for i, row in dataset.iterrows()]

    # print tanimoto matrix
    tanimoto_matrix = np.empty((len(mol_set), len(mol_set)))
    for i in range(len(mol_set)):
        for j in range(len(mol_set)):
                tanimoto_matrix[i][j] = 1 - DataStructs.FingerprintSimilarity(mol_set[i][0], mol_set[j][0])

    # Non-diagonal elements
    nondiag_similarities = 1 - tanimoto_matrix[~np.eye(tanimoto_matrix.shape[0],
                                                       dtype=bool)].reshape(tanimoto_matrix.shape[0], -1)

    print('\nMean & Stdev Tanimoto similarity in dataset: {:.3f} +- {:.3f}'.format(np.mean(nondiag_similarities),
                                                                                    np.std(nondiag_similarities)))
    print('Max Tanimoto similarity in dataset: {:.6f}\n'.format(nondiag_similarities.max()))

    plt.figure(figsize=(7, 7))
    plt.matshow(1-tanimoto_matrix, fignum=1)
    plt.title('Tanimoto Similarity Matrix between original data (sorted by dock score)')
    cb = plt.colorbar(fraction=0.046, pad=0.04)
    plt.savefig('Tanimoto_orig.png')

    # score_set = [mol_set[0][1]]
    # sample_set = [mol_set[0]]

    # Put Enamine molecules in the score set
    score_set = [ mol[1] for mol in mol_set[0:106]]
    sample_set = mol_set[0:106]
    # del mol_set[0]
    # for i in range(args.n-1):
    for i in range(args.n):
        distances = np.empty((len(sample_set),len(mol_set)))
        for j, samp in enumerate(sample_set):
            distances[j] = 1 - np.array([DataStructs.FingerprintSimilarity(samp[0], mol[0]) for mol in mol_set])

        # find minmax index (normal)
        # min_dists = [distances.argmin(axis=1), distances.min(axis=0)]
        # max_dists = [min_dists[1].argmax(axis=0) ,min_dists[1].max(axis=0)]

        min_dists = distances.min(axis=0)
        max_inds = np.flip(np.argsort(min_dists)) # Find indices in descending order
        for ind in max_inds:
            if mol_set[ind][4]=='covid_SA':
                sample_set.append(mol_set[ind])
                score_set.append(mol_set[ind][1])
                del mol_set[ind]
                break

        # append to sample set
        # sample_set.append(mol_set[max_dists[0]])
        # score_set.append(mol_set[max_dists[0]][1])

        # delete from mols
        # del mol_set[max_dists[0]]

        # # find minmax index (favour Enamine)
        # min_dists = [distances.argmin(axis=1), distances.min(axis=0)]
        # inds = np.argpartition(min_dists[1], -3)[-3:] # find top 3
        # inds = np.flip(inds[np.argsort(min_dists[1][inds])]) # sort those 3
        #
        # skip = False
        # for ind in inds:
        #     if mol_set[ind][4]=='enamine':
        #         sample_set.append(mol_set[ind])
        #         del mol_set[ind]
        #         skip = True
        #         break
        # if not skip:
        #     # No enamine molecule, just add the best one
        #     sample_set.append(mol_set[inds[0]])
        #     del mol_set[inds[0]]


    # print tanimoto matrix
    if args.sort:
        sample_set = np.array(sample_set)[np.argsort(np.array(score_set))]

    tanimoto_matrix = np.empty((len(sample_set), len(sample_set)))
    for i in range(len(sample_set)):
        for j in range(len(sample_set)):
                tanimoto_matrix[i][j] = 1 - DataStructs.FingerprintSimilarity(sample_set[i][0], sample_set[j][0])

    # Non-diagonal elements
    nondiag_similarities = 1 - tanimoto_matrix[~np.eye(tanimoto_matrix.shape[0],
                                                       dtype=bool)].reshape(tanimoto_matrix.shape[0], -1)
    plt.clf()
    plt.figure(figsize=(7,7))
    plt.matshow(1-tanimoto_matrix, fignum=1)
    cb = plt.colorbar(fraction=0.046, pad=0.04)
    #plt.show()
    if args.sort:
        plt.title('Tanimoto Similarity Matrix between chosen samples (sorted by score)')
        plt.savefig('Tanimoto_sorted.png')
    else:
        plt.title('Tanimoto Similarity Matrix between chosen samples (unsorted)')
        plt.savefig('Tanimoto.png')

    print('\nMean & Stdev Tanimoto similarity in sample set: {:.3f} +- {:.3f}'.format(np.mean(nondiag_similarities),
                                                                                    np.std(nondiag_similarities)))
    print('Max Tanimoto similarity in sample set: {:.3f}\n'.format(nondiag_similarities.max()))

    # save to .csv
    df = pd.DataFrame(columns=['SMILES', 'CID', 'source', 'dock_score'])
    df['SMILES'] = [samp[2] for samp in sample_set]
    df['CID'] = [samp[3] for samp in sample_set]
    df['dock_score'] = [samp[1] for samp in sample_set]
    df['source'] = [samp[4] for samp in sample_set]

    # enamine_df = pd.read_csv('data/processed_data_enamine.csv', usecols=['SMILES', 'CID', 'source', 'dock_score'])
    # df = pd.concat([df, enamine_df])

    print(df.describe())
    print(df['source'].value_counts())
    df.sort_values(by='dock_score').to_csv('data/optimised_sample_subset_updated.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-csv', type=str, default='data/processed_data.csv',
                        help='path to .csv file of processed_data.')
    parser.add_argument('-n', type=int, default=200,
                        help='number of samples to choose.')
    parser.add_argument('-sort', type=bool, default=False,
                        help='whether or not to sort the sample set before plotting the tanimoto matrix')
    args = parser.parse_args()


    main(args)