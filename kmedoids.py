import argparse

import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.Chem import Draw, MolFromSmiles, AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmilesFromSmiles
import matplotlib.pyplot as plt

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def kMedoids(D, k, tmax=100):
    # determine dimensions of distance matrix D
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')

    # find a set of valid initial cluster medoid indices since we
    # can't seed different clusters with two points at the same location
    valid_medoid_inds = set(range(n))
    invalid_medoid_inds = set([])
    rs, cs = np.where(D == 0)

    # the rows, cols must be shuffled because we will keep the first duplicate below
    index_shuf = list(range(len(rs)))
    np.random.shuffle(index_shuf)
    rs = rs[index_shuf]
    cs = cs[index_shuf]

    for r,c in zip(rs,cs):
        # if there are two points with a distance of 0...
        # keep the first one for cluster init
        if r < c and r not in invalid_medoid_inds:
            invalid_medoid_inds.add(c)
    valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds)

    print('number of valid medoids: {}'.format(len(valid_medoid_inds)))
    if k > len(valid_medoid_inds):
        raise Exception('too many medoids (after removing {} duplicate points)'.format(
            len(invalid_medoid_inds)))

    # randomly initialize an array of k medoid indices
    M = np.array(valid_medoid_inds)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    converge = False
    for t in range(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:, M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J == kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa], C[kappa])], axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            converge = True
            print('\nConvergence reached at t = {}!'.format(t))
            break
        M = np.copy(Mnew)

    if not converge:
        print('\nNot yet converged!')

    # final update of cluster memberships
    J = np.argmin(D[:, M], axis=1)
    for kappa in range(k):
        C[kappa] = np.where(J == kappa)[0]

    # return results
    return M, C


def main(args):
    data = pd.read_csv('data/unique_docked_smiles.csv')
    orig_len = len(data)
    # filter out invalid SMILES
    data['mol'] = data['SMILES'].apply(lambda x: MolFromSmiles(x))
    data = data[data['mol'].notnull()]
    scaff_list = []
    for i, smiles in enumerate(data['SMILES'].values):
        scaff_list.append(MurckoScaffoldSmilesFromSmiles(smiles))
    data['scaffold'] = scaff_list
    new_len = len(data)
    print('\n{} lines have been removed because of invalid SMILES'.format(orig_len-new_len))

    fprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024) for mol in data['mol'].values]

    if args.load_mat:
        D_matrix = np.load('D_mat.npy')
        print(D_matrix)
    else:
        print('\nCalculating Tanimoto matrix...')
        # distance matrix
        D_matrix = np.empty((len(fprints), len(fprints)))
        for i in range(len(fprints)):
            for j in range(len(fprints)):
                    D_matrix[i][j] = 1 - DataStructs.FingerprintSimilarity(fprints[i], fprints[j])
        np.save('D_mat.npy', D_matrix)
        print(D_matrix)

    # Plot Tanimoto distance
    if args.plot:
        plt.figure(figsize=(7, 7))
        plt.matshow(1-D_matrix, fignum=1)
        plt.title('Tanimoto Similarity in dataset')
        cb = plt.colorbar(fraction=0.046, pad=0.04)
        plt.savefig('Tanimoto_sim.png')

        plt.figure(figsize=(7, 7))
        plt.matshow(D_matrix, fignum=1)
        plt.title('Tanimoto Distance in dataset')
        plt.savefig('Tanimoto_dist.png')
    print('\nRunning kmedoids...')
    # split into 15 clusters
    M, C = kMedoids(D_matrix, args.K)

    print('\nmedoids:')
    medoid_list = data.iloc[M]
    smiles_list = [smiles for smiles in medoid_list['SMILES'].values]

    img = Draw.MolsToGridImage(medoid_list['mol'].values, molsPerRow=5, subImgSize=(400,400))
    img.save('data/medoids_grid.png')

    print('\nclustering result:')
    data['cluster'] = 0
    for label in C:
        #print('No. of mols in cluster {} = {}'.format(label, len(C[label])))
        data['cluster'].iloc[C[label]] = label
        # for point_idx in C[label]:
        #    print('label {0}:　{1}'.format(label, data[point_idx]))
    data = data[['SMILES','TITLE','dockvalent_id','cluster']]
    data.to_csv('data/kclustered_smiles.csv', index=False)

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