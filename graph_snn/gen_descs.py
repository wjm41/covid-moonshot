import numpy as np
import pandas as pd
from rdkit import Chem
from mpi4py import MPI
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, mol_to_bigraph
from descriptastorus.descriptors import rdNormalizedDescriptors

def return_borders(index, dat_len, mpi_size):
    mpi_borders = np.linspace(0, dat_len, mpi_size + 1).astype('int')

    border_low = mpi_borders[index]
    border_high = mpi_borders[index+1]
    return border_low, border_high

def generate_descriptors(smi):
    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    features = generator.process(smi)[1:]
    return features[:114]

def main():
    """
    :param n_trials: int specifying number of random train/test splits to use
    :param test_set_size: float in range [0, 1] specifying fraction of dataset to use as test set
    """
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    df = pd.read_csv('data/sars_lip.csv')
    smiles_list = df['smiles']

    my_border_low, my_border_high = return_borders(mpi_rank, len(smiles_list), mpi_size)

    my_smiles = smiles_list[my_border_low:my_border_high]
    my_mols = np.array([Chem.MolFromSmiles(m) for m in my_smiles])

    # Initialise featurisers
    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer()

    e_feats = bond_featurizer.feat_size('e')
    n_feats = atom_featurizer.feat_size('h')

    my_graphs = np.array([mol_to_bigraph(m, node_featurizer=atom_featurizer,
                                         edge_featurizer=bond_featurizer) for m in my_mols])

    sendcounts = np.array(mpi_comm.gather(len(my_graphs), root=0))

    # my_descs = np.array([generate_descriptors(m) for m in my_smiles])
    # if mpi_rank == 0:
        # descs = np.empty((len(smiles_list), 114), dtype=np.float64)
    # else:
        # descs = None

    # mpi_comm.Gatherv(sendbuf=my_descs, recvbuf=(descs, sendcounts), root=0)
    graphs = mpi_comm.gather(my_graphs, root=0)
    X = graphs[0]
    if mpi_rank==0:
        for graph in graphs:
            X = X.vstack([X,graph])
        # np.save('/rds-d2/user/wjm41/hpc-work/sars_descs.npy', descs)
        np.save('/rds-d2/user/wjm41/hpc-work/sars_graphs.npy', X)
        print('SAVED!')

if __name__ == '__main__':
    main()