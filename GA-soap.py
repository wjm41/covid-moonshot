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
from dscribe.kernels import REMatchKernel
from sklearn.preprocessing import normalize

from helper import split_by_lengths
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


def pop_fitness(population, ref_soap, rcut, sigma, max_score=-9999, fit_mean=0, fit_std=1):
    """
    Calculates the fitness (ie SOAP similarity score) of the population by generating conformers for each of the
    population molecules, then evaluating their SOAP descriptors and calculating its similarity score with the SOAP
    descriptor of the binding ligand 'field'

    :param population: list of RDKit molecule objects
    :param ref_soap: numpy array containing the SOAP descriptor of the binding ligand field
    :param rcut, sigma: SOAP parameters
    :param max_score: Maximum SOAP similarity found so far
    :param fit_mean: Mean SOAP similarity of initial population (starts at 0)
    :param fit_std: Stdev of SOAP similarity of initial population

    :return: fitness, max_score, fit_mean, fit_std
    """

    # loop over RDKit mols and turn them into lists of ASE atom objects for dscribe SOAP atomic feature generation
    population_ase = []
    num_list = []
    species = ['C']
    for m in population:
        m = Chem.AddHs(m)
        AllChem.EmbedMolecule(m, maxAttempts=1000)
        m = Chem.RemoveHs(m)
        num_list.append(len(m.GetAtoms()))
        for i, atom in m.GetAtoms():
            symbol = atom.GetSymbol()
            conf = m.GetConformer()
            population_ase.append(Atoms(symbol, [conf.GetPositions()[i]]))
            if symbol not in species:  # find unique atomic species for SOAP generation
                species.append(symbol)

    soap_generator = SOAP(species=species, periodic=False, rcut=rcut, nmax=12, lmax=8, sigma=sigma, sparse=True)
    soap = soap_generator.create(population_ase)

    # normalize SOAP atom descriptors and group by molecule
    soap = normalize(soap, copy=False)
    soap = split_by_lengths(soap, num_list)

    # TODO make REMatch kernel args as input args
    re = REMatchKernel(metric="polynomial", degree=3, gamma=1, coef0=0, alpha=0.5, threshold=1e-6,
                       normalize_kernel=True)

    fitness = re.create(soap, ref_soap)

    # update max_score, include new champion
    if np.amax(fitness) > max_score[0]:
        max_score = [np.amax(fitness), Chem.MolToSmiles(population[np.argmax(fitness)])]

    # normalize fitness to turn them into probability scores
    fitness = np.maximum((fitness - fit_mean) / fit_std, 0.0)

    fitness = fitness / np.sum(fitness)

    return fitness, max_score, fit_mean, fit_std


def initialise_system(args):
    """
    Reads in a .csv file and generates a population of RDKit molecules - initialises max score,
    the mean and stdev fitness scores, and loads in ref_soap

    :param args: system arguments parsed into main - should contain args.csv, args.tgt, args.rcut, and args.sigma

    :return: population, pop_size, pop_fit, pop_mean, pop_std, max_score
    """
    population = []
    csv = pd.read_csv(args.csv, header=None, names=['SMILES'])
    for i, row in csv.iterrows():
        population.append(Chem.MolFromSmiles(row['SMILES']))
    # mols, num_list, atom_list, species = read_xyz(xyz_file) # mols are a list of ASE atoms objects
    pop_size = len(population)

    ref_soap = np.load(args.ref_soap)

    # Need to generate initial fitness here
    pop_fit, max_score, _, _ = pop_fitness(population, ref_soap, args.rcut, args.sigma)

    pop_mean = np.mean(pop_fit)
    pop_std = np.std(pop_fit)

    csv['fitness'] = pop_fit

    print('\nInitial population:')
    print('SMILES: {}, Fitness: {}'.format(csv['SMILES'].values, csv['fitness'].values))

    return population, pop_size, pop_fit, pop_mean, pop_std, max_score, ref_soap


def main(args):
    """
    Runs genetic algorithm - initialise_system generates the starting population and loads in the SOAP descriptors
    of the binding ligand 'field'. Program then loops over generations, calculates the fitness of the population,
    prints+saves the highest fitness individual ('champion'), and then updates the population based on the fitness.

    :param args: system arguments parsed into program
    :return:
    """
    t0 = time.time()

    co.average_size = args.avg_size  # read what this does
    co.size_stdev = args.size_stdev

    population, pop_size, fitness, fit_mean, fit_std, max_score, ref_soap  = initialise_system(args)

    print('\nInitial Population Size: {}'.format(pop_size))
    print('No. of generations: {}'.format(args.n_gens))
    print('Mutation rate: {}'.format(args.mr))
    print('')

    f = open('champions.dat', 'w')
    for generation in range(args.n_gens):
        print('\nGeneration #{}'.format(generation))

        fitness = pop_fitness(population, ref_soap, args.rcut, args.sigma, max_score, fit_mean, fit_std)
        population = reproduce(population, fitness, args.mr)

        # Think you might want to print out the best-k individuals from each generation - wil leave that to you
        print('Champion fitness = {} with {}'.format(max_score[0], max_score[1]))
        f.write(max_score[1] + '\t' + str(max_score[0]) + '\n')
        f.flush()

    t1 = time.time()
    print('\nTime taken: {}'.format(t1 - t0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-csv', type=str,
                        help='path to .csv file of initial population.')
    parser.add_argument('-tgt', type=str,
                        help='path to .npz file containing SOAP descriptors of the binding fragments.')
    parser.add_argument('--mut_rate', '-mr', type=float, default=0.01,
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
    args = parser.parse_args()

    main(args)
