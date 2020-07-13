# Author: Arian Jamasb
"""
Property prediction using a Message-Passing Neural Network.
"""

import argparse
import logging


import dgl
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from mpnn import MPNNPairPredictorMulti
from parser import return_score_pairs

logging.basicConfig(level=logging.INFO)
if torch.cuda.is_available():
    print('use GPU')
    device = 'cuda'
else:
    print('use CPU')
    device = 'cpu'

def return_borders(index, dat_len, size):
    borders = np.linspace(0, dat_len, size + 1).astype('int')

    border_low = borders[index]
    border_high = borders[index+1]
    return border_low, border_high

# Collate Function for Dataloader
def collate(sample):
    graphs_high, graphs_low = map(list, zip(*sample))
    batched_graph_high = dgl.batch(graphs_high)
    batched_graph_high.set_n_initializer(dgl.init.zero_initializer)
    batched_graph_high.set_e_initializer(dgl.init.zero_initializer)
    batched_graph_low = dgl.batch(graphs_low)
    batched_graph_low.set_n_initializer(dgl.init.zero_initializer)
    batched_graph_low.set_e_initializer(dgl.init.zero_initializer)
    return batched_graph_high, batched_graph_low

def main(args):
    """
    :param n_trials: int specifying number of random train/test splits to use
    :param test_set_size: float in range [0, 1] specifying fraction of dataset to use as test set
    """

    df_targets = pd.read_csv('data/'+args.input)
    df_bmarks = pd.read_csv('data/'+args.target+'_hits.csv')

    index = int(args.index)
    mpi_size = int(args.size)
    len_lib = len(df_targets)
    border_low, border_high = return_borders(index, len_lib, size=mpi_size)
    df_targets = df_targets[border_low:border_high]
    preds = []

    for i in tqdm(range(args.n_batches)):
        logging.info('Scoring batch #{}'.format(i))
        new_len = len(df_targets)
        border_low, border_high = return_borders(i, new_len, size=args.n_batches)
        batched_df = df_targets[border_low:border_high]

        X_high, X_low, n_feats, e_feats = return_score_pairs(batched_df, df_bmarks)

        data = list(zip(X_high, X_low))

        data_loader = DataLoader(data, batch_size=len(df_bmarks), collate_fn=collate, drop_last=False)

        n_tasks = 1
        mpnn_net = MPNNPairPredictorMulti(node_in_feats=n_feats,
                                       edge_in_feats=e_feats,
                                       node_out_feats=128,
                                       n_tasks=n_tasks)
        mpnn_net.load_state_dict(torch.load('/rds-d2/user/wjm41/hpc-work/models/'+args.modelname+'/model_epoch_final.pt'))
        mpnn_net = mpnn_net.to(device)

        mpnn_net.eval()

        for i, (bg_high, bg_low) in enumerate(data_loader):
            atom_feats_high = bg_high.ndata.pop('h').to(device)
            bond_feats_high = bg_high.edata.pop('e').to(device)
            atom_feats_low = bg_low.ndata.pop('h').to(device)
            bond_feats_low = bg_low.edata.pop('e').to(device)
            with torch.no_grad():
                y_pred = mpnn_net(bg_high, atom_feats_high, bond_feats_high, bg_low, atom_feats_low, bond_feats_low)
            y_pred = F.sigmoid(y_pred)

            y_pred = y_pred.detach().cpu().numpy()

            # store labels and preds
            preds.append(y_pred)

    preds = np.array(preds).squeeze()

    #np.savetxt(args.target+'_scores.txt', preds)
    df_targets['avg_score'] = np.mean(preds, axis=1)
    df_targets.to_csv(args.savename+'_scores_batch_'+str(index)+'.csv', index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-n_batches', type=int, default=10,
                        help='int specifying number of batches for scoring')
    parser.add_argument('-modelname', type=str, default='multitask_pair',
                        help='name for directory containing saved model params and tensorboard logs')
    parser.add_argument('-savename', type=str, default='multitask_pair',
                        help='name for directory containing saved model params and tensorboard logs')
    parser.add_argument('-index', type=str, default='0',
                        help='integer index for data batching')
    parser.add_argument('-size', type=str, default='10',
                        help='Number of batches in total.')
    parser.add_argument('-target', type=str, default='acry',
                        help='target series for scoring hits')
    parser.add_argument('-input', type=str,
                        help='input file of smiles to score relative to the targets.')
    parser.add_argument('-dry', action='store_true',
                        help='whether or not to only use a subset of the HTS screen')
    parser.add_argument('-debug', action='store_true',
                        help='whether or not to print predictions and model weight gradients')
    args = parser.parse_args()

    main(args)
