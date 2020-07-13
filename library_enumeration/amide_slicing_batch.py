from rdkit.Chem import AllChem, MolFromSmiles, MolToSmiles, Draw, Crippen
from rdkit.Chem.rdmolops import FastFindRings
from rdkit import Chem
from tqdm import tqdm
import pandas as pd
import numpy as np
from mpi4py import MPI

import logging
from pympler.asizeof import asizeof
from hurry.filesize import size
import sys

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()
if mpi_rank==0:
    print("\nSlicing amide library on " + str(mpi_size) + " MPI processes.\n")


logging.basicConfig(level=logging.INFO)
def canonicalize(smi_list, showprogress=False):
    mol_list = []
    if showprogress:
        print('Canonicalising mols')
        for smi in tqdm(smi_list):
            mol = MolFromSmiles(smi)
            if mol is not None:
                mol_list.append(MolToSmiles(mol))
    else:
        for smi in smi_list:
            mol = MolFromSmiles(smi)
            if mol is not None:
                mol_list.append(mol)
    mol_list = list(set(mol_list))
    final_list = []
    if showprogress:
        print('Size of unfiltered final library: {}'.format(len(mol_list)))
        print('Filtering by n_heavy and logP:')
        for smi in tqdm(mol_list):
            mol = MolFromSmiles(smi)
            n_heavy = mol.GetNumHeavyAtoms()
            if n_heavy > 17:
                logP = Crippen.MolLogP(mol)
                if logP <= 5:
                    final_list.append(smi)
    else:
        for smi in mol_list:
            mol = MolFromSmiles(smi)
            n_heavy = mol.GetNumHeavyAtoms()
            if n_heavy > 17:
                logP = Crippen.MolLogP(mol)
                if logP <= 5:
                    final_list.append(smi)
    return final_list

def pair_rxnts(mol1_list, mol2_list, rxn, debug=False):
    prod_list = []
    for mol1 in mol1_list:
        for mol2 in mol2_list:
            products = rxn.RunReactants((Chem.AddHs(mol1),Chem.AddHs(mol2)))
            if debug:
                logging.info(products)
            if products != ():
                for prod in products:
                    if debug:
                        logging.info(MolToSmiles(prod[0]))
                    prod_list.append(MolToSmiles(prod[0]))
    return prod_list

def return_borders(index, dat_len, size):
    borders = np.linspace(0, dat_len, size + 1).astype('int')

    border_low = borders[index]
    border_high = borders[index+1]
    return border_low, border_high

extra = AllChem.ReactionFromSmarts('[N:1][n,c:2].[N,O,C;!$(NC=O):3][c:4]>>[*:3][*:2]')

file = open('amine_list.txt', 'r')
amine_list = file.read().splitlines()
amine_list = [MolFromSmiles(smi) for smi in amine_list]
if mpi_rank==0:
    print('Finised reading amines')

file2 = open('penul_lib.txt', 'r')
penultimate_lib = file2.read().splitlines()
if mpi_rank==0:
    print('Finised reading penultimates')

index = int(sys.argv[1])
job_size = 10
len_lib = len(penultimate_lib)
border_low, border_high = return_borders(index, len_lib, size=job_size)
penultimate_lib = penultimate_lib[border_low:border_high]
my_low, my_high = return_borders(mpi_rank, len(penultimate_lib), size=mpi_size)
penultimate_lib = penultimate_lib[my_low:my_high]
penultimate_lib_mols = [MolFromSmiles(smi) for smi in penultimate_lib]
print('Finished converting to RDKit Mols')

extra_mols = pair_rxnts(amine_list, penultimate_lib_mols, extra)
if mpi_rank==0:
    print('Finished running reactions')
    print('Number of extra mols and their size in memory: {},{}'.format(len(extra_mols),size(sys.getsizeof(extra_mols))))
final_lib = canonicalize(penultimate_lib + extra_mols, showprogress=True)
if mpi_rank==0:
    print('Finished canonicalisation')
file = open('noncovalent_lib_final_batch_'+str(index)+'_rank_'+str(mpi_rank)+'.txt', 'w')
file.write('SMILES\n')
file.writelines("%s\n" % mol for mol in final_lib)