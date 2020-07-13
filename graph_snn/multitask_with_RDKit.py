# Author: Arian Jamasb
"""
Property prediction using a Message-Passing Neural Network.
"""

import argparse
import os

import dgl
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from mpnn import CustomMPNNPredictor
from descriptastorus.descriptors import rdNormalizedDescriptors
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, mol_to_bigraph
from rdkit import Chem
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, auc, precision_recall_curve
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch import nn
from torch.nn import functional as F
from torch.nn import MSELoss, BCELoss
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

#Set torch variables
torch.autograd.set_detect_anomaly(True)

if torch.cuda.is_available():
    print('use GPU')
    device = 'cuda'
else:
    print('use CPU')
    device = 'cpu'

class Net(nn.Module):
    """Simple container for transformimg MPNN outputs to 3 classifications and 3 regressions
    Parameters
    ----------

    """
    def __init__(self, class_inds, reg_inds):
        super(Net, self).__init__()
        self.class_inds = class_inds
        self.reg_inds = reg_inds

    def forward(self, preds):
        """
        """
        outputs = []
        for ind in self.class_inds:
            outputs.append(F.softmax(preds[:,ind],dim=0))
        for ind in self.reg_inds:
            outputs.append(preds[:,ind])
        return torch.stack(outputs).T

def generate_descriptors(smi):
    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    features = generator.process(smi)[1:]
    return features[:114]

# Collate Function for Dataloader
def collate(sample):
    graphs, descs, labels = map(list, zip(*sample))
    batched_graph = dgl.batch(graphs)
    batched_graph.set_n_initializer(dgl.init.zero_initializer)
    batched_graph.set_e_initializer(dgl.init.zero_initializer)
    descs = torch.cat(descs).reshape(len(labels),114)
    labels = torch.cat(labels).reshape(len(labels),6)
    return batched_graph, descs, labels

def main(args):
    """
    :param n_trials: int specifying number of random train/test splits to use
    :param test_set_size: float in range [0, 1] specifying fraction of dataset to use as test set
    """

    df = pd.read_csv('data/covid_multitask_pIC50.smi')
    smiles_list = df['SMILES'].values
    y = df[['acry_class', 'chloro_class', 'rest_class', 'acry_reg', 'chloro_reg', 'rest_reg']].to_numpy()

    n_tasks = y.shape[1]
    class_inds = [0,1,2]
    reg_inds = [3,4,5]
    X = [Chem.MolFromSmiles(m) for m in smiles_list]
    descs = np.array([generate_descriptors(m) for m in smiles_list])
    # Initialise featurisers
    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer()

    e_feats = bond_featurizer.feat_size('e')
    n_feats = atom_featurizer.feat_size('h')
    print('Number of features: ', n_feats)

    X = np.array([mol_to_bigraph(m, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer) for m in X])

    r2_list = []
    rmse_list = []
    roc_list = []
    prc_list = []

    for i in range(args.n_trials):
        writer = SummaryWriter('runs/'+args.savename+'/run_' + str(i))

        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_set_size, random_state=i+5)
        if args.test:
            X_train_acry, X_test_acry, \
            descs_train_acry, descs_test_acry, \
            y_train_acry, y_test_acry = train_test_split(X[~np.isnan(y[:,0])], descs[~np.isnan(y[:, 0])],
                                                         y[~np.isnan(y[:,0])], stratify=y[:,0][~np.isnan(y[:,0])],
                                                         test_size=args.test_set_size, shuffle=True, random_state=i+5)
            X_train_chloro, X_test_chloro, \
            descs_train_chloro, descs_test_chloro, \
            y_train_chloro, y_test_chloro = train_test_split(X[~np.isnan(y[:,1])], descs[~np.isnan(y[:, 1])],
                                                             y[~np.isnan(y[:,1])], stratify=y[:,1][~np.isnan(y[:,1])],
                                                              test_size=args.test_set_size, shuffle=True, random_state=i+5)
            X_train_rest, X_test_rest, \
            descs_train_rest, descs_test_rest, \
            y_train_rest, y_test_rest = train_test_split(X[~np.isnan(y[:,2])], descs[~np.isnan(y[:, 2])],
                                                         y[~np.isnan(y[:,2])], stratify=y[:,2][~np.isnan(y[:,2])],
                                                          test_size=args.test_set_size, shuffle=True, random_state=i+5)

            X_train = np.concatenate([X_train_acry, X_train_chloro, X_train_rest])
            X_test = np.concatenate([X_test_acry, X_test_chloro, X_test_rest])
            desc_train = np.concatenate([descs_train_acry, descs_train_chloro, descs_train_rest])
            desc_test = np.concatenate([descs_test_acry, descs_test_chloro, descs_test_rest])
            y_train = np.concatenate([y_train_acry, y_train_chloro, y_train_rest])
            y_test = np.concatenate([y_test_acry, y_test_chloro, y_test_rest])

            desc_train = torch.Tensor(desc_train)
            desc_test = torch.Tensor(desc_test)
            y_train = torch.Tensor(y_train)
            y_test = torch.Tensor(y_test)

            train_data = list(zip(X_train, desc_train, y_train))
            test_data = list(zip(X_test, desc_test, y_test))

            train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate, drop_last=False)
            test_loader = DataLoader(test_data, batch_size=32, shuffle=True, collate_fn=collate, drop_last=False)
        else:
            train_data = list(zip(X, descs, y))
            train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate, drop_last=False)

        process = Net(class_inds, reg_inds)
        process = process.to(device)

        mpnn_net = CustomMPNNPredictor(node_in_feats=n_feats,
                                       edge_in_feats=e_feats,
                                       node_out_feats=128,
                                       n_tasks=n_tasks)
        mpnn_net = mpnn_net.to(device)

        reg_loss_fn = MSELoss()
        class_loss_fn = BCELoss()

        optimizer = torch.optim.Adam(mpnn_net.parameters(), lr=0.001)

        for epoch in range(1, args.n_epochs+1):
            epoch_loss = 0
            preds = []
            labs = []
            mpnn_net.train()
            n=0
            for i, (bg, dcs, labels) in enumerate(train_loader):
                dcs = dcs.to(device)
                labels = labels.to(device)
                atom_feats = bg.ndata.pop('h').to(device)
                bond_feats = bg.edata.pop('e').to(device)
                y_pred = mpnn_net(bg, atom_feats, bond_feats, dcs)
                y_pred = process(y_pred)
                if args.debug:
                    print('y_pred: {}'.format(y_pred))
                    print('label: {}'.format(labels))
                loss = torch.tensor(0)
                loss = loss.to(device)
                for ind in reg_inds:
                    if len(labels[:,ind][~torch.isnan(labels[:,ind])])==0:
                        continue
                    loss = loss + reg_loss_fn(y_pred[:,ind][~torch.isnan(labels[:,ind])],
                                              labels[:,ind][~torch.isnan(labels[:,ind])])
                if args.debug:
                    print('reg loss: {}'.format(loss))

                for ind in class_inds:
                    if len(labels[:,ind][~torch.isnan(labels[:,ind])])==0:
                        continue
                    loss = loss + class_loss_fn(y_pred[:,ind][~torch.isnan(labels[:,ind])],
                                                labels[:,ind][~torch.isnan(labels[:,ind])])

                optimizer.zero_grad()
                loss.backward()
                if args.debug:
                    print('class + reg loss: {}'.format(loss))
                    print('y_pred: {}'.format(y_pred))
                    print('label: {}'.format(labels))
                # for p in mpnn_net.parameters():
                #     print(p.grad)
                optimizer.step()
                if args.debug:
                    n+=1
                    if n==10:
                        raise Exception
                epoch_loss += loss.detach().item()

                labels = labels.cpu().numpy()
                y_pred = y_pred.detach().cpu().numpy()

                # store labels and preds
                preds.append(y_pred)
                labs.append(labels)

            labs = np.concatenate(labs, axis=0)
            preds = np.concatenate(preds, axis=0)
            rmses= []
            r2s = []
            rocs = []
            prcs = []
            for ind in reg_inds:
                rmse = np.sqrt(mean_squared_error(labs[:,ind][~np.isnan(labs[:,ind])],
                                                  preds[:,ind][~np.isnan(labs[:,ind])]))
                r2 = r2_score(labs[:,ind][~np.isnan(labs[:,ind])],
                              preds[:,ind][~np.isnan(labs[:,ind])])
                rmses.append(rmse)
                r2s.append(r2)

            for ind in class_inds:
                roc = roc_auc_score(labs[:,ind][~np.isnan(labs[:,ind])],
                                   preds[:,ind][~np.isnan(labs[:,ind])])
                precision, recall, thresholds = precision_recall_curve(labs[:,ind][~np.isnan(labs[:,ind])],
                                                                       preds[:,ind][~np.isnan(labs[:,ind])])
                prc = auc(recall, precision)
                rocs.append(roc)
                prcs.append(prc)

            writer.add_scalar('LOSS/train', epoch_loss, epoch)
            writer.add_scalar('train/acry_rocauc', rocs[0], epoch)
            writer.add_scalar('train/acry_prcauc', prcs[0], epoch)
            writer.add_scalar('train/chloro_rocauc', rocs[1], epoch)
            writer.add_scalar('train/chloro_prcauc', prcs[1], epoch)
            writer.add_scalar('train/rest_rocauc', rocs[2], epoch)
            writer.add_scalar('train/rest_prcauc', prcs[2], epoch)

            writer.add_scalar('train/acry_rmse', rmses[0], epoch)
            writer.add_scalar('train/acry_r2', r2s[0], epoch)
            writer.add_scalar('train/chloro_rmse', rmses[1], epoch)
            writer.add_scalar('train/chloro_r2', r2s[1], epoch)
            writer.add_scalar('train/rest_rmse', rmses[2], epoch)
            writer.add_scalar('train/rest_r2', r2s[2], epoch)

            if epoch % 20 == 0:
                print(f"\nepoch: {epoch}, "
                      f"LOSS: {epoch_loss:.3f}"
                      f"\n acry ROC-AUC: {rocs[0]:.3f}, "
                      f"acry PRC-AUC: {prcs[0]:.3f}"
                      f"\n chloro ROC-AUC: {rocs[1]:.3f}, "
                      f"chloro PRC-AUC: {prcs[1]:.3f}"
                      f"\n rest ROC-AUC: {rocs[2]:.3f}, "
                      f"rest PRC-AUC: {prcs[2]:.3f}"
                      f"\n acry R2: {r2s[0]:.3f}, "
                      f"acry RMSE: {rmses[0]:.3f}"
                      f"\n chloro R2: {r2s[1]:.3f}, "
                      f"chloro RMSE: {rmses[1]:.3f}"
                      f"\n rest R2: {r2s[2]:.3f}, "
                      f"rest RMSE: {rmses[2]:.3f}")
                try:
                    torch.save(mpnn_net.state_dict(), 'models/' + args.savename + '/model_epoch_' + str(epoch) + '.pt')
                except FileNotFoundError:
                    cmd = 'mkdir models/' + args.savename
                    os.system(cmd)
                    torch.save(mpnn_net.state_dict(),
                               'models/' + args.savename + '/model_epoch_' + str(epoch) + '.pt')
            if args.test:
                # Evaluate
                mpnn_net.eval()
                preds = []
                labs = []
                for i, (bg, dcs, labels) in enumerate(test_loader):
                    dcs = dcs.to(device)
                    labels = labels.to(device)
                    atom_feats = bg.ndata.pop('h').to(device)
                    bond_feats = bg.edata.pop('e').to(device)
                    y_pred = mpnn_net(bg, atom_feats, bond_feats, dcs)
                    y_pred = process(y_pred)

                    labels = labels.cpu().numpy()
                    y_pred = y_pred.detach().cpu().numpy()

                    preds.append(y_pred)
                    labs.append(labels)

                labs = np.concatenate(labs, axis=0)
                preds = np.concatenate(preds, axis=0)
                rmses = []
                r2s = []
                rocs = []
                prcs = []
                for ind in reg_inds:

                    rmse = np.sqrt(mean_squared_error(labs[:,ind][~np.isnan(labs[:,ind])],
                                                      preds[:,ind][~np.isnan(labs[:,ind])]))

                    r2 = r2_score(labs[:,ind][~np.isnan(labs[:,ind])],
                                  preds[:,ind][~np.isnan(labs[:,ind])])
                    rmses.append(rmse)
                    r2s.append(r2)
                for ind in class_inds:
                    roc = roc_auc_score(labs[:, ind][~np.isnan(labs[:,ind])],
                                       preds[:, ind][~np.isnan(labs[:,ind])])
                    precision, recall, thresholds = precision_recall_curve(labs[:, ind][~np.isnan(labs[:,ind])],
                                                                           preds[:, ind][~np.isnan(labs[:,ind])])
                    prc = auc(recall, precision)
                    rocs.append(roc)
                    prcs.append(prc)
                writer.add_scalar('test/acry_rocauc', rocs[0], epoch)
                writer.add_scalar('test/acry_prcauc', prcs[0], epoch)
                writer.add_scalar('test/chloro_rocauc', rocs[1], epoch)
                writer.add_scalar('test/chloro_prcauc', prcs[1], epoch)
                writer.add_scalar('test/rest_rocauc', rocs[2], epoch)
                writer.add_scalar('test/rest_prcauc', prcs[2], epoch)

                writer.add_scalar('test/acry_rmse', rmses[0], epoch)
                writer.add_scalar('test/acry_r2', r2s[0], epoch)
                writer.add_scalar('test/chloro_rmse', rmses[1], epoch)
                writer.add_scalar('test/chloro_r2', r2s[1], epoch)
                writer.add_scalar('test/rest_rmse', rmses[2], epoch)
                writer.add_scalar('test/rest_r2', r2s[2], epoch)
                if epoch==(args.n_epochs):
                    print(f"\n======================== TEST ========================"
                          f"\n acry ROC-AUC: {rocs[0]:.3f}, "
                          f"acry PRC-AUC: {prcs[0]:.3f}"
                          f"\n chloro ROC-AUC: {rocs[1]:.3f}, "
                          f"chloro PRC-AUC: {prcs[1]:.3f}"
                          f"\n rest ROC-AUC: {rocs[2]:.3f}, "
                          f"rest PRC-AUC: {prcs[2]:.3f}"
                          f"\n acry R2: {r2s[0]:.3f}, "
                          f"acry RMSE: {rmses[0]:.3f}"
                          f"\n chloro R2: {r2s[1]:.3f}, "
                          f"chloro RMSE: {rmses[1]:.3f}"
                          f"\n rest R2: {r2s[2]:.3f}, "
                          f"rest RMSE: {rmses[2]:.3f}")
                    roc_list.append(rocs)
                    prc_list.append(prcs)
                    r2_list.append(r2s)
                    rmse_list.append(rmses)

    roc_list = np.array(roc_list).T
    prc_list = np.array(prc_list).T
    r2_list = np.array(r2_list).T
    rmse_list = np.array(rmse_list).T
    print("\n ACRY")
    print("R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list[0]), np.std(r2_list[0])/np.sqrt(len(r2_list[0]))))
    print("RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list[0]), np.std(rmse_list[0])/np.sqrt(len(rmse_list[0]))))
    print("ROC-AUC: {:.3f} +- {:.3f}".format(np.mean(roc_list[0]), np.std(roc_list[0]) / np.sqrt(len(roc_list[0]))))
    print("PRC-AUC: {:.3f} +- {:.3f}".format(np.mean(prc_list[0]), np.std(prc_list[0]) / np.sqrt(len(prc_list[0]))))
    print("\n CHLORO")
    print("R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list[1]), np.std(r2_list[1])/np.sqrt(len(r2_list[1]))))
    print("RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list[1]), np.std(rmse_list[1])/np.sqrt(len(rmse_list[1]))))
    print("ROC-AUC: {:.3f} +- {:.3f}".format(np.mean(roc_list[1]), np.std(roc_list[1]) / np.sqrt(len(roc_list[1]))))
    print("PRC-AUC: {:.3f} +- {:.3f}".format(np.mean(prc_list[1]), np.std(prc_list[1]) / np.sqrt(len(prc_list[1]))))
    print("\n REST")
    print("R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list[2]), np.std(r2_list[2])/np.sqrt(len(r2_list[2]))))
    print("RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list[2]), np.std(rmse_list[2])/np.sqrt(len(rmse_list[2]))))
    print("ROC-AUC: {:.3f} +- {:.3f}".format(np.mean(roc_list[2]), np.std(roc_list[2]) / np.sqrt(len(roc_list[2]))))
    print("PRC-AUC: {:.3f} +- {:.3f}".format(np.mean(prc_list[2]), np.std(prc_list[2]) / np.sqrt(len(prc_list[2]))))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-n_trials', '--n_trials', type=int, default=3,
                        help='int specifying number of random train/test splits to use')
    parser.add_argument('-n_epochs', type=int, default=200,
                        help='int specifying number of epochs for training')
    parser.add_argument('-savename', '--savename', type=str, default='multitask_full',
                        help='name for directory containing saved model params and tensorboard logs')
    parser.add_argument('-ts', '--test_set_size', type=float, default=0.2,
                        help='float in range [0, 1] specifying fraction of dataset to use as test set')
    parser.add_argument('-debug', action='store_true',
                        help='whether or not to print tensor values')
    parser.add_argument('-test', action='store_true',
                        help='whether or not to do train/test split')
    args = parser.parse_args()

    main(args)
