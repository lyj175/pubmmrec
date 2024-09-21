from __future__ import division
from __future__ import print_function

import argparse
import time
from queue import Queue

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import coo_matrix
from torch import optim, nn
from sklearn.decomposition import TruncatedSVD

from src.models.gae.model import GCNModelVAE
from src.models.gae.optimizer import loss_function
from src.models.gae.utils import load_data, mask_test_edges, preprocess_graph, get_roc_score

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')

args = parser.parse_args()


def id_map(data,ori_feature):
    index_query_feature_id = {}
    feature_id_query_index = {}
    features = []
    data = list(data)
    for i in range(0, len(data)):
        index_query_feature_id[i] = data[i]
        feature_id_query_index[data[i]] = i
        features.append(ori_feature[i].clone())
    return index_query_feature_id, feature_id_query_index, features
def iter_node(root_index, until,node_interaction_dict):
    node_queue = Queue()
    node_queue.put(root_index)
    rows, cols, keys = [], [], []
    while node_queue.qsize() > 0:
        cur = node_queue.get()
        children = node_interaction_dict[cur].tolist()[:10]  # TODO 只取前十个交互结点
        rows += [cur] * len(children)
        cols += children
        keys = list(keys)
        keys += [cur] * len(children)
        keys += children
        keys = set(keys)

        if len(keys) >= until: return rows, cols, keys
        for i in children:
            node_queue.put(i)
def get_norm_index(row,col,id_to_index_dict):
    new_row,new_col = [],[]
    for i in range(0,len(row)):
        new_row.append(id_to_index_dict[row[i]])
        new_col.append(id_to_index_dict[col[i]])
    return new_row,new_col
def item_modal_forward(features,num_user,node_interaction_dict,root):
        # TODO 样本特征从item扩展到user
        num_user = num_user
        feature_users = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
            np.random.randn(num_user, features.shape[1]), dtype=torch.float32, requires_grad=True)))
        # feature_users, features = feature_users.to(self.device), features.to(self.device)
        features_new = torch.cat((feature_users, features), dim=0)

        subgraph_data_row,subgraph_data_col,all_count = iter_node(root,2000,node_interaction_dict)
        # subgraph_data_row,subgraph_data_col,all_count = iter_node(root,1000,node_interaction_dict)

        index_query_feature_id,feature_id_query_index,features_subgraph = id_map(all_count,features_new)
        features_subgraph = torch.stack(features_subgraph)
        print(f'去重前{len(subgraph_data_row)}去重后{len(set(all_count))}')

        subgraph_data_row,subgraph_data_col = get_norm_index(subgraph_data_row,subgraph_data_col,feature_id_query_index)
        new_coo_matrix = coo_matrix((np.array([1] * len(subgraph_data_row)), (subgraph_data_row, subgraph_data_col)),
                                    shape=(len(all_count), len(all_count)))
        features_sub_graph,new_adj = features_subgraph,new_coo_matrix
        # return forward(features_sub_graph,new_adj,adj,features)
        return features_sub_graph,new_adj


def gae_for(args,fea,num_user,node_dict):
    print("Using {} dataset".format(args.dataset_str))
    # adj, features = load_data(args.dataset_str)
    n_nodes, feat_dim = fea.shape

    # features,adj = item_modal_forward(fea,num_user,node_dict,)
    # n_nodes = features.shape[0]
    #
    # # Store original adjacency matrix (without diagonal entries) for later
    # adj_orig = adj
    # adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    # adj_orig.eliminate_zeros()
    #
    # adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    # adj = adj_train
    #
    # # Some preprocessing
    # adj_norm = preprocess_graph(adj)
    # adj_label = adj_train + sp.eye(adj_train.shape[0])
    # # adj_label = sparse_to_tuple(adj_label)
    # adj_label = torch.FloatTensor(adj_label.toarray())
    #
    # pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    # norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    #TODO args.hidden1：基础编码输出层大小。args.hidden2:方差，均值计算层的输出大小
    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    hidden_emb = None
    count = 0
    for epoch in range(args.epochs):
        t = time.time()
        features, adj = item_modal_forward(fea, num_user, node_dict, count)
        # features, adj = item_modal_forward(fea, num_user, node_dict, 20000)
        count += 1
        n_nodes = features.shape[0]

        # Store original adjacency matrix (without diagonal entries) for later
        adj_orig = adj
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()

        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
        adj = adj_train

        # Some preprocessing
        adj_norm = preprocess_graph(adj)
        adj_label = adj_train + sp.eye(adj_train.shape[0])
        # adj_label = sparse_to_tuple(adj_label)
        adj_label = torch.FloatTensor(adj_label.toarray())

        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

        model.train()
        optimizer.zero_grad()
        recovered, mu, logvar = model(features, adj_norm)
        loss = loss_function(preds=recovered, labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight)
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        hidden_emb = mu.data.numpy()
        roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
              "val_ap=", "{:.5f}".format(ap_curr),
              "time=", "{:.5f}".format(time.time() - t)
              )

    print("Optimization Finished!")

    roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))


if __name__ == '__main__':
    gae_for(args)
