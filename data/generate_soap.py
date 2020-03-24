"""Generates the SOAP descriptors for the binding ligand field"""
import argparse

import numpy as np
from dscribe.descriptors import SOAP
from sklearn.preprocessing import normalize

from helper import read_xyz

def main(args):
    """
    Generates SOAP descriptors for the atoms saved in args.xyz
    :param args:
    :return:
    """
    mols, num_list, atom_list, species = read_xyz(args.xyz)

    soap_generator = SOAP(species=species, periodic=False, rcut=args.rcut, nmax=8, lmax=6, sigma=args.sigma, sparse=True)

    soap = soap_generator.create(mols)

    soap = normalize(soap, copy=False)

    np.save(args.tgt, [soap])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-xyz', type=str,
                        help='path to .xyz file of ligand atom coordinates.')
    parser.add_argument('-tgt', type=str,
                        help='path to output .npy file of ligand SOAP descriptors.')
    parser.add_argument('-rcut', type=float, default=3.0,
                        help='rcut for SOAP feature generation.')
    parser.add_argument('-sigma', type=float, default=0.2,
                        help='sigma for SOAP feature generation.')
    args = parser.parse_args()

    main(args)
