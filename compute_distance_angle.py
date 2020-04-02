import argparse

import numpy as np
from scipy.spatial.distance import euclidean as euc_dist
import copy

from helper import read_xyz

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def dist_angle_calculator(u_frag, u_overlap, v_frag, v_overlap, u_connections = 1, v_connections = 1, verbose=True):
    u_frag = read_xyz(u_frag)[0].get_positions()
    u_overlap = read_xyz(u_overlap)[0].get_positions()
    v_frag = read_xyz(v_frag)[0].get_positions()
    v_overlap = read_xyz(v_overlap)[0].get_positions()

    u_exits, u_linkeds = coordinate_finder(u_frag, u_overlap, u_connections, verbose)
    v_exits, v_linkeds = coordinate_finder(v_frag, v_overlap, v_connections, verbose)

    distances = np.zeros((u_exits.shape[0], v_exits.shape[0]))
    angles = np.zeros((u_exits.shape[0], v_exits.shape[0]))
    for i, u_exit in enumerate(u_exits):
        u_linked = u_linkeds[i, :]
        for j, v_exit in enumerate(v_exits):
            v_linked = v_linkeds[j, :]
            distances[i, j], angles[i, j] = distance_angle(u_exit, u_linked, v_exit, v_linked)
    return distances, angles


def distance_angle(u_exit, u_linked, v_exit, v_linked):
    distance = euc_dist(u_exit, v_exit)
    # Get angle
    v1_u = unit_vector(u_exit - u_linked)
    v2_u = unit_vector(v_exit - v_linked)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    return distance, angle


def coordinate_finder(u_frag, u_overlap, u_connections, verbose = True):
    #Calculate the coordinates of u_exit
    min_dist = 3.0*np.ones(u_connections+1)
    u_exits = np.zeros((u_connections+1, 3))

    for i, u in enumerate(u_frag):
        for j, v in enumerate(u_overlap):
            dist = euc_dist(u, v)
            min_dist[-1] = dist

            u_exits[-1, :] = copy.deepcopy(u)
            idx_sort = np.argsort(min_dist)
            min_dist = min_dist[idx_sort]
            u_exits = u_exits[idx_sort, :]

    u_exits = u_exits[:-1, :]
    u_linkeds = np.zeros((u_connections, 3))
    for n, u_exit in enumerate(u_exits):
        min_dist2 = 3.0
        for i, u in enumerate(u_frag):
            dist = euc_dist(u, u_exit)
            if (0.05 < dist < min_dist2):
                min_dist2 = dist
                u_linked = copy.deepcopy(u)
                idx_linked = i
        u_linkeds[n, :] = u_linked

    if verbose == True:
        print(u_exits)
        print(u_linkeds)
    return u_exits, u_linkeds


def main(args):
    """
    Loops over covalent and non-covalent fragments, finds the pairs that have planar overlap in 3D space, then writes
    the pairs to args.out
    """

    print('Beginning analysis...')

    path = 'data/overlaps/' + args.covalent + '_' + args.non_covalent + '/'
    u_frag = path + args.covalent + '_frag.xyz'
    u_overlap = path + args.covalent + '_overlap.xyz'
    v_frag = path + args.non_covalent + '_frag.xyz'
    v_overlap = path + args.non_covalent + '_overlap.xyz'

    distances, angles = dist_angle_calculator(u_frag, u_overlap, v_frag, v_overlap,
                                              u_connections = args.cov_connections,
                                              v_connections = args.non_cov_connections)
    print(distances, angles)

    f = open(path + args.covalent + '_' + args.non_covalent + '_distances_angles.xyz', 'w')
    f.write("Distances = " + str(distances) + '\n')
    f.write("Angles = " + str(angles) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-covalent', type=str, default='0692',
                        help='Covalent fragment indices')
    parser.add_argument('-non_covalent', type=str, default='0072',
                        help='Non-covalent fragment indices')
    parser.add_argument('-cov_connections', type=int, default=1,
                        help='Number of attachments to the overlapping structure (i.e. number of fragments)')
    parser.add_argument('-non_cov_connections', type=int, default=1,
                        help='Number of attachments to the overlapping structure (i.e. number of fragments)')
    parser.add_argument('-n_overlap',type=int, default=4,
                        help='minimum number of atom pairs within threshold for fragment pair to count as overlapping')
    parser.add_argument('-verbose', type=bool, default=False,
                        help='whether to print extra statements')
    args = parser.parse_args()

    main(args)