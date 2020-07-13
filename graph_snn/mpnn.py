# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# MPNN
# pylint: disable= no-member, arguments-differ, invalid-name
import torch
import torch.nn as nn

from dgl.nn.pytorch import Set2Set
from dgllife.model.gnn import MPNNGNN

__all__ = ['CustomMPNNPredictor', 'MPNN_encoder', 'MPNNPairPredictor','MPNNPairPredictorMulti']

class MPNN_encoder(nn.Module):
    """MPNN encoder for regression and classification on graphs.
    MPNN is introduced in `Neural Message Passing for Quantum Chemistry
    <https://arxiv.org/abs/1704.01212>`__.
    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    edge_in_feats : int
        Size for the input edge features.
    node_out_feats : int
        Size for the output node representations. Default to 64.
    edge_hidden_feats : int
        Size for the hidden edge representations. Default to 128.
    num_step_message_passing : int
        Number of message passing steps. Default to 6.
    num_step_set2set : int
        Number of set2set steps. Default to 6.
    num_layer_set2set : int
        Number of set2set layers. Default to 3.
    """
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats=64,
                 edge_hidden_feats=128,
                 num_step_message_passing=6,
                 num_step_set2set=6,
                 num_layer_set2set=3):
        super(MPNN_encoder, self).__init__()

        self.gnn = MPNNGNN(node_in_feats=node_in_feats,
                           node_out_feats=node_out_feats,
                           edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing)
        self.readout = Set2Set(input_dim=node_out_feats,
                               n_iters=num_step_set2set,
                               n_layers=num_layer_set2set)
        self.process = nn.Sequential(
            # nn.Linear(2 * node_out_feats + 114, node_out_feats),
            nn.Linear(2 * node_out_feats, node_out_feats),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
    def forward(self, g, node_feats, edge_feats, descs=0):
        """Graph-level regression/soft classification.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features.
        descs: float32 tensor of shape (G, 115)
            Input RDKit descriptors
        Returns
        -------
        float32 tensor of shape (G, n_tasks)
            Learnt graph-level features. G for the number of graphs.
        """
        node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)
        # graph_feats = torch.cat((graph_feats,descs), dim=1) # concatenate graph features with RDKit features
        return self.process(graph_feats)

# pylint: disable=W0221
class CustomMPNNPredictor(nn.Module):
    """
    Parameters
    ----------
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    """
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats=64,
                 edge_hidden_feats=128,
                 n_tasks=1,
                 num_step_message_passing=6,
                 num_step_set2set=6,
                 num_layer_set2set=3):
        super(CustomMPNNPredictor, self).__init__()

        self.encoder = MPNN_encoder(node_in_feats=node_in_feats,
                                   node_out_feats=node_out_feats,
                                   edge_in_feats=edge_in_feats,
                                   edge_hidden_feats=edge_hidden_feats,
                                   num_step_message_passing=num_step_message_passing,
                                   num_step_set2set=num_step_set2set,
                                   num_layer_set2set=num_layer_set2set
        )
        self.predict = nn.Sequential(
            nn.Linear(node_out_feats, node_out_feats),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(node_out_feats, node_out_feats),
            nn.ReLU(),
            nn.Dropout(0,2),
            nn.Linear(node_out_feats, n_tasks)
        )
    def encode(self,g,node_feats, edge_feats):
        return self.encoder(g, node_feats,edge_feats)

    def forward(self, g, node_feats, edge_feats, descs=0):
        """Graph-level regression/soft classification.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features.
        descs: float32 tensor of shape (G, 115)
            Input RDKit descriptors
        Returns
        -------
        float32 tensor of shape (G, n_tasks)
            Prediction for the graphs in the batch. G for the number of graphs.
        """
        graph_feats = self.encode(g, node_feats, edge_feats, descs)
        # graph_feats = self.encode(g, node_feats, edge_feats)
        return self.predict(graph_feats)


class MPNNPairPredictor(nn.Module):
    """
    Parameters
    ----------
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    """

    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats=64,
                 edge_hidden_feats=128,
                 n_tasks=1,
                 num_step_message_passing=6,
                 num_step_set2set=6,
                 num_layer_set2set=3):
        super(MPNNPairPredictor, self).__init__()

        self.encoder = MPNN_encoder(node_in_feats=node_in_feats,
                                    node_out_feats=node_out_feats,
                                    edge_in_feats=edge_in_feats,
                                    edge_hidden_feats=edge_hidden_feats,
                                    num_step_message_passing=num_step_message_passing,
                                    num_step_set2set=num_step_set2set,
                                    num_layer_set2set=num_layer_set2set
                                    )
        self.predict = nn.Sequential(
            nn.Linear(node_out_feats, node_out_feats),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(node_out_feats, node_out_feats),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(node_out_feats, n_tasks),
        )

    def encode(self, g, node_feats, edge_feats):
        return self.encoder(g, node_feats, edge_feats)

    def forward(self, g1, nodes_1, edges_1, g2, nodes_2, edges_2):
        """Graph-level regression/soft classification.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features.
        descs: float32 tensor of shape (G, 115)
            Input RDKit descriptors
        Returns
        -------
        float32 tensor of shape (G, n_tasks)
            Prediction for the graphs in the batch. G for the number of graphs.
        """
        graph_1 = self.encode(g1, nodes_1, edges_1)
        graph_2 = self.encode(g2, nodes_2, edges_2)
        graph_diff = graph_1 - graph_2
        return self.predict(graph_diff)

class MPNNPairPredictorMulti(nn.Module):
    """
    Parameters
    ----------
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    """

    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats=64,
                 edge_hidden_feats=128,
                 n_tasks=1,
                 num_step_message_passing=6,
                 num_step_set2set=6,
                 num_layer_set2set=3):
        super(MPNNPairPredictorMulti, self).__init__()

        self.encoder = MPNN_encoder(node_in_feats=node_in_feats,
                                    node_out_feats=node_out_feats,
                                    edge_in_feats=edge_in_feats,
                                    edge_hidden_feats=edge_hidden_feats,
                                    num_step_message_passing=num_step_message_passing,
                                    num_step_set2set=num_step_set2set,
                                    num_layer_set2set=num_layer_set2set
                                    )
        self.predict = nn.Sequential(
            nn.Linear(node_out_feats, node_out_feats),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(node_out_feats, node_out_feats),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(node_out_feats, n_tasks),
        )

    def encode(self, g, node_feats, edge_feats):
        return self.encoder(g, node_feats, edge_feats)

    def forward(self, g1, nodes_1, edges_1, g2, nodes_2, edges_2):
        """Graph-level regression/soft classification.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features.
        descs: float32 tensor of shape (G, 115)
            Input RDKit descriptors
        Returns
        -------
        float32 tensor of shape (G, n_tasks)
            Prediction for the graphs in the batch. G for the number of graphs.
        """
        graph_1 = self.encode(g1, nodes_1, edges_1)
        graph_2 = self.encode(g2, nodes_2, edges_2)
        graph_diff = graph_1 - graph_2
        return self.predict(graph_diff)