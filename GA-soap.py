"""Runs graph-based genetic algorithm to optimize the SOAP similarity between initial population and target data"""

import time
import random
import argparse

import numpy as np
from ase import Atoms
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from dscribe.descriptors import SOAP
from dscribe.kernels import REMatchKernel, AverageKernel
from sklearn.preprocessing import normalize

from helper import read_xyz, split_by_lengths
# The following two are written by Jensen
import crossover as co
import mutate as mu


def reproduce(population, fitness, mutation_rate):
    """
    Generates next generation of population by probabilistically choosing mating pool based on fitness, then
    probabilistically reproducing molecules in the mating pool and randomly mutating the children

    :param population: list of RDKit molecules
    :param fitness: probability distribution of same length as population, returned by pop_fitness
    :param mutation_rate: hyperparameter determining the likelihood of mutations occuring

    :return: new_population
    """
    mating_pool = []
    for i in range(len(population)):
        mating_pool.append(np.random.choice(population, p=fitness))

    new_population = []
    for n in range(len(population)):
        parent_A = random.choice(mating_pool)
        parent_B = random.choice(mating_pool)
        # print Chem.MolToSmiles(parent_A),Chem.MolToSmiles(parent_B)
        new_child = co.crossover(parent_A, parent_B)
        # print new_child
        if new_child is not None:
            new_child = mu.mutate(new_child, mutation_rate)
            # print("after mutation",new_child)
            if new_child is not None:
                new_population.append(new_child)

    return new_population


def pop_fitness(population, rcut, sigma, kernel, tgt_atoms, tgt_species, max_score=[-9999,'']):
    """
    Calculates the fitness (ie SOAP similarity score) of the population by generating conformers for each of the
    population molecules, then evaluating their SOAP descriptors and calculating its similarity score with the SOAP
    descriptor of the binding ligand 'field'

    Conformer generation and similarity calculation are the computational bottlenecks  - might be worth splitting the
    task up with MPI. see return_borders.py in helper.py if you want to do that - make sure you only run the
    reproduction on the master node (since there is randomness), then broadcast to the other nodes

    :param population: list of RDKit molecule objects
    :param tgt_atoms: list of ASE atom objects of the target ligand field - from read_xyz
    :param tgt_species: list of the atomic species present in the target ligand field - from read_xyz
    :param rcut, sigma: SOAP parameters
    :param max_score: Maximum SOAP similarity found so far

    :return: fitness, max_score, fit_mean, fit_std
    """
    t0 = time.time()

    # loop over RDKit mols and turn them into lists of ASE atom objects for dscribe SOAP atomic feature generation
    population_ase = []
    num_list = []
    species = ['C']
    bad_mols = []
    for m in population:
        m = Chem.AddHs(m)
        conf_result = AllChem.EmbedMolecule(m, maxAttempts=1000)
        if conf_result != 0:
            bad_mols.append(m)
            continue
        m = Chem.RemoveHs(m)
        num_list.append(len(m.GetAtoms()))
        for i, atom in enumerate(m.GetAtoms()):
            symbol = atom.GetSymbol()
            conf = m.GetConformer()
            population_ase.append(Atoms(symbol, [conf.GetPositions()[i]]))
            if symbol not in species:  # find unique atomic species for SOAP generation
                species.append(symbol)
    population.remove(bad_mols) # filter out molecules which have no conformers

    # Check that we also include the atom types present in the ligand targets
    for atom in tgt_species:
        if atom not in species:
            species.append(atom)
    t1 = time.time()
    print('Time taken to generate conformers: {}'.format(t1-t0))

    # Generate SOAP descriptors using dscribe
    soap_generator = SOAP(species=species, periodic=False, rcut=rcut, nmax=8, lmax=6, sigma=sigma, sparse=True)
    soap = soap_generator.create(population_ase)
    tgt_soap = soap_generator.create(tgt_atoms)

    # normalize SOAP atom descriptors and group by molecule
    soap = normalize(soap, copy=False)
    tgt_soap = [normalize(tgt_soap, copy=False)]
    soap = split_by_lengths(soap, num_list)

    t2 = time.time()
    print('Time taken to generate SOAP descriptors: {}'.format(t2-t1))

    # TODO make REMatch kernel args as input args
    if kernel == 'rematch:':
        soap_similarity = REMatchKernel(metric="polynomial", degree=3, gamma=1, coef0=0, alpha=0.1, threshold=1e-3,
                                        normalize_kernel=True)
    elif kernel == 'average':
        soap_similarity = AverageKernel(metric="polynomial", degree=3, gamma=1, coef0=0, normalize_kernel=True)
    fitness = soap_similarity.create(soap, tgt_soap)
    fitness = fitness.flatten()

    t3 = time.time()
    print('Time taken to calculate fitness: {}'.format(t3-t2))
    # update max_score, include new champion
    if np.amax(fitness) > max_score[0]:
        max_score = [np.amax(fitness), Chem.MolToSmiles(population[np.argmax(fitness)])]

    # normalize fitness to turn them into probability scores
    fitness = fitness / np.sum(fitness)

    return fitness, max_score


def initialise_system(args):
    """
    Reads in a .csv file and generates a population of RDKit molecules, as well as reading in target ligand coordinates

    :param args: system arguments parsed into main - should contain args.csv, args.tgt

    :return: population, tgt_atoms, tg_species
    """
    population = []
    csv = pd.read_csv(args.csv, header=0)
    for i, row in csv.iterrows():
        population.append(Chem.MolFromSmiles(row['SMILES']))

    tgt_atoms, _, _, tgt_species = read_xyz(args.tgt)

    return population, tgt_atoms, tgt_species


def main(args):
    """
    Runs genetic algorithm - initialise_system generates the starting population. Program then loops over generations,
    calculates the fitness of the population, prints+saves the highest fitness individual ('champion'),
    and then updates the population based on the fitness.

    :param args: system arguments parsed into program
    :return:
    """
    t0 = time.time()

    co.average_size = args.tgt_size  # read what this does
    co.size_stdev = args.size_stdev

    population, tgt_atoms, tgt_species = initialise_system(args)

    print('\nInitial Population Size: {}'.format(len(population)))
    print('No. of generations: {}'.format(args.n_gens))
    print('Mutation rate: {}'.format(args.mut_rate))
    print('')

    max_score = [-999, '']
    f = open('data/champions.dat', 'w')
    for generation in range(args.n_gens):
        print('\nGeneration #{}, population size: {}'.format(generation, len(population)))
        print('Calculating fitness...')
        fitness, max_score = pop_fitness(population, args.rcut, args.sigma, args.kernel,
                                         tgt_atoms, tgt_species, max_score)
        print('Producing next generation...')
        population = reproduce(population, fitness, args.mut_rate)

        # Think you might want to print out the best-k individuals from each generation - wil leave that to you
        print('Champion fitness = {}, smiles = {}'.format(max_score[0], max_score[1]))
        f.write(max_score[1] + '\t' + str(max_score[0]) + '\n')
        f.flush()

    t1 = time.time()
    print('\nTime taken: {}'.format(t1 - t0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-csv', type=str, default='data/covid_submissions.csv',
                        help='path to .csv file of initial population.')
    parser.add_argument('-tgt', type=str, default='data/xyz/all_ligands.xyz',
                        help='path to .xyz file containing binding fragments coordinates.')
    parser.add_argument('-mut_rate', type=float, default=0.01,
                        help='Probability of mutations.')
    parser.add_argument('-n_gens', type=int, default=50,
                        help='Number of generations to evolve the population.')
    parser.add_argument('-tgt_size', type=float, default=39.15,
                        help='Molecule size used in molecule crossover - should be the size of the target molecule.')
    parser.add_argument('-size_stdev', type=float, default=3.50,
                        help='Stdev of molecule crossover.')
    parser.add_argument('-rcut', type=float, default=3.0,
                        help='rcut for SOAP feature generation.')
    parser.add_argument('-sigma', type=float, default=0.2,
                        help='sigma for SOAP feature generation.')
    parser.add_argument('-kernel', type=str, default='average',
                        help='SOAP kernel used for similarity score - either "average" or "rematch"')
    args = parser.parse_args()

    main(args)
