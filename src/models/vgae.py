import argparse
import random
import time
from queue import Queue

import numpy as np
import torch
from scipy.sparse import coo_matrix
from torch import nn, optim

from src.models.gae.optimizer import loss_function
from src.models.gae.utils import mask_test_edges, preprocess_graph, get_roc_score
from src.models.gae.model import GCNModelVAE
import scipy.sparse as sp


class VGAE():
    def __init__(self,num_user,num_item,node_dict,fea,dim):
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
        self.num_user = num_user
        self.num_item = num_item
        self.node_dict = node_dict
        self.fea = fea

        # TODO args.hidden1：基础编码输出层大小。args.hidden2:方差，均值计算层的输出大小
        # self.model = GCNModelVAE(fea.shape[1], args.hidden1, args.hidden2, args.dropout)
        self.model = GCNModelVAE(dim, dim, dim, args.dropout)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        self.args = args

    def id_map(self,data, ori_feature):
        index_query_feature_id = {}
        feature_id_query_index = {}
        features = []
        data = list(data)
        for i in range(0, len(data)):
            index_query_feature_id[i] = data[i]
            feature_id_query_index[data[i]] = i
            features.append(ori_feature[data[i]].clone())
        return index_query_feature_id, feature_id_query_index, features

    def iter_node(self,root_index, until, node_interaction_dict):
        node_queue = Queue()
        node_queue.put(root_index)
        rows, cols, keys = [], [], []
        while node_queue.qsize() > 0:
            cur = node_queue.get()
            children = node_interaction_dict[cur].tolist()[:8]  # TODO 只取前十个交互结点
            rows += [cur] * len(children)
            cols += children
            keys = list(keys)
            keys += [cur] * len(children)
            keys += children
            keys = set(keys)
            # print(b,'-----------',cur,'----------------',len(rows),'----',len(cols),'-------',len(keys),'------------',until)
            if len(cols) >= until: return rows, cols, keys
            for i in children:
                node_queue.put(i)

    def get_norm_index(self,row, col, id_to_index_dict):
        new_row, new_col = [], []
        for i in range(0, len(row)):
            new_row.append(id_to_index_dict[row[i]])
            new_col.append(id_to_index_dict[col[i]])
        return new_row, new_col

    #TODO 生成子图
    def item_modal_forward(self,features, num_user, node_interaction_dict, root):
        # TODO 样本特征从item扩展到user
        # num_user = num_user
        # feature_users = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
        #     np.random.randn(num_user, features.shape[1]), dtype=torch.float32, requires_grad=True)))
        # # feature_users, features = feature_users.to(self.device), features.to(self.device)
        # features_new = torch.cat((feature_users, features), dim=0)
        #all_count记录样本索引，不重复
        #TODO 图结点数最大值，在第一次迭代后开始判断
        subgraph_data_row, subgraph_data_col, all_count = self.iter_node(root, 1, node_interaction_dict)
        # subgraph_data_row,subgraph_data_col,all_count = iter_node(root,1000,node_interaction_dict)

        index_query_feature_id, feature_id_query_index, features_subgraph = self.id_map(all_count, features)
        features_subgraph = torch.stack(features_subgraph)
        # print(f'去重前{len(subgraph_data_row)}去重后{len(set(all_count))}')

        subgraph_data_row, subgraph_data_col = self.get_norm_index(subgraph_data_row, subgraph_data_col,
                                                              feature_id_query_index)
        new_coo_matrix = coo_matrix((np.array([1] * len(subgraph_data_row)), (subgraph_data_row, subgraph_data_col)),
                                    shape=(len(all_count), len(all_count)))
        features_sub_graph, new_adj = features_subgraph, new_coo_matrix
        # return forward(features_sub_graph,new_adj,adj,features)
        return features_sub_graph, new_adj

    def set_fea(self,fea):
        self.fea = fea

    #TODO 生成n个子图用作正例，返回原样本与增强样本
    def simulate(self,n):
        # n_nodes, feat_dim = fea.shape
        # hidden_emb = None
        # for epoch in range(args.epochs):
        t = time.time()
        n_index = []
        while len(n_index)<n:
            index = int(random.random()*(self.num_user+self.num_item))
            if index not in n_index: n_index.append(index)
        # print('n个样本的索引',n_index)
        results = [[],[]]#TODO 存放原始图0索引与vgae生成图

        for i in n_index:#TODO 对随机抽取的中心节点开始往外扩张，得到抽取的子图
            features, adj = self.item_modal_forward(self.fea, self.num_user, self.node_dict,i)
            # features, adj = item_modal_forward(fea, num_user, node_dict, 20000)
            n_nodes = features.shape[0]

            # Store original adjacency matrix (without diagonal entries) for later
            adj_orig = adj
            adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
            adj_orig.eliminate_zeros()

            adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
            if adj_train == None:
                continue
            adj = adj_train

            # Some preprocessing
            adj_norm = preprocess_graph(adj)
            adj_label = adj_train + sp.eye(adj_train.shape[0])
            # adj_label = sparse_to_tuple(adj_label)
            adj_label = torch.FloatTensor(adj_label.toarray())

            if adj.sum()==0:continue;
            pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
            norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

            # self.model.train()
            # self.optimizer.zero_grad()
            recovered, mu, logvar,z = self.model(features, adj_norm)
            loss = loss_function(preds=recovered, labels=adj_label,
                                 mu=mu, logvar=logvar, n_nodes=n_nodes,
                                 norm=norm, pos_weight=pos_weight)
            results[0].append(features)
            results[1].append(z)
            # loss.backward(retain_graph=True)
            # cur_loss = loss.item()
            # self.optimizer.step()

            # hidden_emb = mu.data.numpy()
            # roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)

            # print("VGAE----样本:", '%04d' % (i), "train_loss=",
            #       "共%04d"%len(n_index),
            #       "{:.5f}".format(loss),
            #       "val_ap=", "{:.5f}".format(ap_curr),
            #       "time=", "{:.5f}".format(time.time() - t))
            # print("train_loss=", "{:.5f}".format(cur_loss),
            #       "val_ap=", "{:.5f}".format(ap_curr),
            #       "time=", "{:.5f}".format(time.time() - t)
            #       )

            # roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
            # print('Test ROC score: ' + str(roc_score))
            # print('Test AP score: ' + str(ap_score))
        # return results,cur_loss
        return results,loss
