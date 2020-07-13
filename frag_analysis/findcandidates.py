"""Short script for outputting covalent-noncovalent fragment pairs that overlap in space"""

import argparse
import subprocess
import os

import numpy as np
from scipy.spatial.distance import euclidean as euc_dist
import matplotlib.pyplot as plt

from helper import read_xyz

def return_overlap(x1, x2, radius=0.8, n_overlap=3, make_plot=False, verbose=False):
    """
    Calculates the Euclidean distance matrix between 2 fragments, and returns True or False depending on whether or not
    the fragments overlap in space, as determined by a threshold on the Euclidean distance.

    :param x1, x2: .xyz files of the fragment atom coordinates
    :param radius: threshold for determining atom overlap (in angstroms)
    :param n_overlap: number of overlapping atoms for determining fragment pair overlap
    :param make_plot: set to True to plot the euclidean distance matrix
    :param verbose: set to True to print the number of overlapping atoms for each x1, x2
    :return:
    """

    # read 2 xyz files
    u_atoms = read_xyz(x1)[0]
    v_atoms = read_xyz(x2)[0]

    # loop over atom xyz vectors, calculate euclidean distance
    dist_mat = np.empty((len(u_atoms),len(v_atoms)))
    for i, u in enumerate(u_atoms.get_positions()):
        for j,v in enumerate(v_atoms.get_positions()):
            dist_mat[i,j] = euc_dist(u, v)

    if make_plot:
        plt.figure(figsize=(7,7))
        plt.matshow(dist_mat, fignum=1)
        cb = plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('Euclidean Distance Matrix')
        plt.savefig('euc_dist.png')

    # apply distance, return True or False for overlap
    overlap_atoms = np.where(np.less_equal(dist_mat,radius),1,0)
    num_overlap_atoms = np.sum(overlap_atoms)

    overlap_atom_indices = np.argwhere(np.less_equal(dist_mat,radius)).T

    if num_overlap_atoms>=n_overlap: # choose n_overlap=4 for planar overlap
        u_overlaps =  u_atoms[overlap_atom_indices[0]]
        v_overlaps = v_atoms[overlap_atom_indices[1]]

        del u_atoms[overlap_atom_indices[0]]
        del v_atoms[overlap_atom_indices[1]]

        if verbose:
            print('Overlap found! Number of overlapping atoms: {}'.format(num_overlap_atoms))
        return True, u_atoms, v_atoms, u_overlaps, v_overlaps
    else:
        if verbose:
            print('No overlap found :(')
        return False, u_atoms, v_atoms, None, None

def main(args):
    """
    Loops over covalent and non-covalent fragments, finds the pairs that have planar overlap in 3D space, then writes
    the pairs to args.out
    """
    print('Beginning analysis...')

    f1 = open(args.cov_inds, 'r')
    covalent_indices = f1.read().splitlines()

    f2 = open(args.ncov_inds, 'r')
    non_covalent_indices = f2.read().splitlines()

    candidate_file = open(args.out, 'w')
    candidate_file.write('covalent,non_covalent\n')
    n_overlaps = 0
    for u in covalent_indices:
        for v in non_covalent_indices:
            overlap, u_atoms, v_atoms, u_overlaps, v_overlaps = return_overlap('data/covalent/Mpro-x'+u+'_0.xyz',
                                                                               'data/non_covalent/Mpro-x'+v+'_0.xyz',
                                                                               radius=args.radius,
                                                                               n_overlap=args.n_overlap,
                                                                               verbose=args.verbose)
            if overlap:
                candidate_file.write(u+','+v+'\n')
                n_overlaps+=1
                if args.write_atoms:

                    # copy original .xyz files
                    os.system('mkdir data/overlaps/'+u+'_'+v)
                    os.system('cp data/covalent/Mpro-x'+u+'_0.xyz data/overlaps/'+u+'_'+v+'/'+u+'_orig.xyz')
                    os.system('cp data/non_covalent/Mpro-x'+v+'_0.xyz data/overlaps/'+u+'_'+v+'/'+v+'_orig.xyz')

                    # record pair indices
                    f = open('data/overlaps/indices.csv', 'a')
                    f.write(u+','+v+'\n')

                    # save coordinates of post-deletion fragments
                    f = open('data/overlaps/'+u+'_'+v+'/'+u+'_frag.xyz', 'w')
                    f.write(str(len(u_atoms))+'\n'*2)
                    u_symbols = u_atoms.get_chemical_symbols()
                    u_xyz = u_atoms.get_positions()
                    for i, atom in enumerate(u_atoms):
                        f.write(u_symbols[i]+'\t'+str(u_xyz[i][0])+'\t'+str(u_xyz[i][1])+'\t'+str(u_xyz[i][2])+'\n')
                    f = open('data/overlaps/' + u + '_' + v + '/' + v + '_frag.xyz', 'w')
                    f.write(str(len(v_atoms)) + '\n' * 2)
                    v_symbols = v_atoms.get_chemical_symbols()
                    v_xyz = v_atoms.get_positions()
                    for i, atom in enumerate(v_atoms):
                        f.write(v_symbols[i] + '\t' + str(v_xyz[i][0])+'\t' + str(v_xyz[i][1]) + '\t' + str(v_xyz[i][2]) + '\n')

                    # save coordinates of overlapping parts
                    f = open('data/overlaps/'+u+'_'+v+'/'+u+'_overlap.xyz', 'w')
                    f.write(str(len(u_overlaps))+'\n'*2)
                    u_symbols = u_overlaps.get_chemical_symbols()
                    u_xyz = u_overlaps.get_positions()
                    for i, atom in enumerate(u_overlaps):
                        f.write(u_symbols[i]+'\t'+str(u_xyz[i][0])+'\t'+str(u_xyz[i][1])+'\t'+str(u_xyz[i][2])+'\n')
                    f = open('data/overlaps/' + u + '_' + v + '/' + v + '_overlap.xyz', 'w')
                    f.write(str(len(v_overlaps)) + '\n' * 2)
                    v_symbols = v_overlaps.get_chemical_symbols()
                    v_xyz = v_overlaps.get_positions()
                    for i, atom in enumerate(v_overlaps):
                        f.write(v_symbols[i] + '\t' + str(v_xyz[i][0])+'\t' + str(v_xyz[i][1]) + '\t' + str(v_xyz[i][2]) + '\n')

    print('Analysis complete: {} overlapping fragments found. Results saved in: {}'.format(n_overlaps, args.out))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cov_inds', type=str, default='data/covalent/indices.txt',
                        help='path to .txt of covalent fragment indices')
    parser.add_argument('-ncov_inds', type=str, default='data/non_covalent/indices.txt',
                        help='path to .txt of non-covalent fragment indices')
    parser.add_argument('-out', type=str, default='data/overlapping_fragments.csv',
                        help='path to output .csv of overlapping fragment pairs')
    parser.add_argument('-r','--radius', type=float, default=0.8,
                        help='radius threshold for determining overlap of fragment atoms')
    parser.add_argument('-n_overlap',type=int, default=4,
                        help='minimum number of atom pairs within threshold for fragment pair to count as overlapping')
    parser.add_argument('-write_atoms', type=bool, default=False,
                        help='whether to write fragment .xyz files with overlapping structure deleted.')
    parser.add_argument('-verbose', type=bool, default=False,
                        help='whether to print extra statements')
    args = parser.parse_args()

    main(args)
