"""Runs graph-based genetic algorithm to optimize the SOAP similarity between initial population and target data"""

import time
import random
import argparse

from mpi4py import MPI
import numpy as np
from ase import Atoms
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from dscribe.descriptors import SOAP
from dscribe.kernels import REMatchKernel, AverageKernel
from sklearn.preprocessing import normalize

from helper import read_xyz, split_by_lengths, return_borders
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


def pop_fitness(mpi_comm, mpi_rank, mpi_size, population, rcut, sigma, kernel, tgt_atoms, tgt_species, tgt_atoms2=None, max_score=[-9999,'']):
    """
    Calculates the fitness (ie SOAP similarity score) of the population by generating conformers for each of the
    population molecules, then evaluating their SOAP descriptors and calculating its similarity score with the SOAP
    descriptor of the binding ligand 'field'

    :param population: list of RDKit molecule objects
    :param tgt_atoms: list of ASE atom objects of the target ligand field - from read_xyz, second is optional if separate sites
    :param tgt_species: list of the atomic species present in the target ligand field - from read_xyz
    :param rcut, sigma: SOAP parameters
    :param max_score: Maximum SOAP similarity found so far

    :return: fitness, max_score, fit_mean, fit_std
    """
    t0 = time.time()

    # partition the population between the MPI cpus
    my_border_low, my_border_high = return_borders(mpi_rank, len(population), mpi_size)
    my_pop = population[my_border_low: my_border_high]
 
    # loop over RDKit mols and turn them into lists of ASE atom objects for dscribe SOAP atomic feature generation
    population_ase = []
    num_list = []
    species = ['C']
    bad_mols = []
    for ind, m in enumerate(my_pop):
        m = Chem.AddHs(m)
        conf_result = AllChem.EmbedMolecule(m, maxAttempts=1000)
        m = Chem.RemoveHs(m)
        num_list.append(len(m.GetAtoms()))
        for i, atom in enumerate(m.GetAtoms()): # this is actually wrong, should have an Atoms object for each mol...
            symbol = atom.GetSymbol()
            if conf_result != 0:
                bad_mols.append(ind)
                population_ase.append(Atoms(symbol, [[0,0,0]]))
            else:
                conf = m.GetConformer()
                population_ase.append(Atoms(symbol, [conf.GetPositions()[i]]))
            if symbol not in species:  # find unique atomic species for SOAP generation
                species.append(symbol)

    # Check that we also include the atom types present in the ligand targets
    for atom in tgt_species:
        if atom not in species:
            species.append(atom)

    t1 = time.time()
    if mpi_rank==0:
        print('Time taken to generate conformers: {}'.format(t1-t0))

    # Generate SOAP descriptors using dscribe
    soap_generator = SOAP(species=species, periodic=False, rcut=rcut, nmax=8, lmax=6, sigma=sigma, sparse=True)
    soap = soap_generator.create(population_ase)
    tgt_soap = soap_generator.create(tgt_atoms)
    if tgt_atoms2 is not None:
        tgt_soap2 = [normalize(soap_generator.create(tgt_atoms2), copy=False)]

    # normalize SOAP atom descriptors and group by molecule
    soap = normalize(soap, copy=False)
    tgt_soap = [normalize(tgt_soap, copy=False)]
    soap = split_by_lengths(soap, num_list)

    t2 = time.time()
    if mpi_rank==0:
        print('Time taken to generate SOAP descriptors: {}'.format(t2-t1))

    # TODO make REMatch kernel args as input args
    if kernel == 'rematch':
        soap_similarity = REMatchKernel(metric="polynomial", degree=3, gamma=1, coef0=0, alpha=0.1, threshold=1e-3, normalize_kernel=True)
    elif kernel == 'average':
        soap_similarity = AverageKernel(metric="polynomial", degree=3, gamma=1, coef0=0, normalize_kernel=True)
    if tgt_atoms2 is not None:
        fitness1 = soap_similarity.create(soap, tgt_soap)
        fitness1.flatten()
        fitness2 = soap_similarity.create(soap, tgt_soap2)
        fitness2.flatten()
        # calculate fitness score as product of the two fitnesses
        fitness = np.multiply(fitness1, fitness2)
        fitness = np.array([f[0] for f in fitness])
    else:
        fitness = soap_similarity.create(soap, tgt_soap)
        fitness = fitness.flatten()

    fitness[bad_mols]=0 # set fitness of bad conformers to 0
    
    sendcounts = np.array(mpi_comm.gather(len(fitness),root=0))

    if mpi_rank==0:
        fitness_full = np.empty(len(population))
    else:
        fitness_full = None

    # Gather fitness arrays from MPI cpus into the root cpu, then broadcast the gathered array to all cpus
    mpi_comm.Gatherv(sendbuf=fitness,recvbuf = (fitness_full, sendcounts),root=0)
    fitness = mpi_comm.bcast(fitness_full, root=0)

    t3 = time.time()
    if mpi_rank==0:
        print('Time taken to calculate fitness: {}'.format(t3-t2))

    # update max_score, include new champion
    if np.amax(fitness) > max_score[0]:
        max_score = [np.amax(fitness), Chem.MolToSmiles(population[np.argmax(fitness)])]
    

    #Print the top 5 scores and corresponding molecules for a particular generation
    top_scores = np.flip(fitness[np.argsort(fitness)[-5:]])
    # print(top_scores)
    for i in range(5):
        if mpi_rank==0:
            print("Mol {}: {} (fitness = {:.3f})".format(i, Chem.MolToSmiles(population[np.argsort(fitness)[-i-1]]), top_scores[i]))
    

    fitness = fitness / np.sum(fitness)

    return fitness, max_score


def initialise_system(args):
    """
    Reads in a .csv file and generates a population of RDKit molecules, as well as reading in target ligand coordinates

    :param args: system arguments parsed into main - should contain args.csv, args.tgt (and args.tgt2)

    :return: population, tgt_atoms, tg_species
     or population, tgt_atoms, tgt_species, tgt_atoms2, tgt_species2
    """
    population = []
    csv = pd.read_csv(args.csv, header=0)
    for i, row in csv.iterrows():
        population.append(Chem.MolFromSmiles(row['SMILES']))

    tgt_atoms, _, _, tgt_species = read_xyz(args.tgt)
    if args.tgt2 is not None:
        tgt_atoms2, _, _, tgt_species2 = read_xyz(args.tgt2)
        tgt_species = list(set().union(tgt_species, tgt_species2)) # creates a single tgt_species list
        return population, tgt_atoms, tgt_species, tgt_atoms2
    else:
        return population, tgt_atoms, tgt_species


def main(args):
    """
    Runs genetic algorithm - initialise_system generates the starting population. Program then loops over generations,
    calculates the fitness of the population, prints+saves the highest fitness individual ('champion'),
    and then updates the population based on the fitness.

    :param args: system arguments parsed into program
    :return:
    """
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    co.average_size = args.tgt_size  # read what this does
    co.size_stdev = args.size_stdev

    if args. tgt2 is not None:
        population, tgt_atoms, tgt_species, tgt_atoms2 = initialise_system(args)
    else:
        population, tgt_atoms, tgt_species = initialise_system(args)
    if mpi_rank==0:
        print('\nInitial Population Size: {}'.format(len(population)))
        print('No. of generations: {}'.format(args.n_gens))
        print('Mutation rate: {}'.format(args.mut_rate))
        print('')

    max_score = [-999, '']
    f = open('champions.dat', 'w')

    for generation in range(args.n_gens):
        if mpi_rank==0:
            print('\nGeneration #{}, population size: {}'.format(generation, len(population)))
            print('Calculating fitness...')
        if args.tgt2 is not None:
            fitness, max_score = pop_fitness(mpi_comm, mpi_rank, mpi_size,population, args.rcut, args.sigma, args.kernel, tgt_atoms, tgt_species, tgt_atoms2, max_score)
        else:
            fitness, max_score = pop_fitness(mpi_comm, mpi_rank, mpi_size, population, args.rcut, args.sigma, args.kernel, tgt_atoms, tgt_species, None,  max_score)
        if mpi_rank==0:
            population = reproduce(population, fitness, args.mut_rate)
        population = mpi_comm.bcast(population, root=0)

        # Think you might want to print out the best-k individuals from each generation - wil leave that to you
        if mpi_rank==0:
            print('Champion fitness = {}, smiles = {}'.format(max_score[0], max_score[1]))
            f.write(max_score[1] + '\t' + str(max_score[0]) + '\n')
            f.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-csv', type=str, default='data/covid_submissions.csv',
                        help='path to .csv file of initial population.')
    parser.add_argument('-tgt', type=str, default='data/xyz/all_ligands.xyz',
                        help='path to .xyz file containing binding fragments coordinates.')
    parser.add_argument('-tgt2', type=str, default=None, 
                        help="path to .xyz file of fragments of second site")
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
