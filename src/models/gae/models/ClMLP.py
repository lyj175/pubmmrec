import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.gae import model as gae_model
from src.models.gae.utils import load_data, mask_test_edges, preprocess_graph, get_roc_score
import scipy.sparse as sp
from src.models.gae.optimizer import loss_function
from scipy.sparse import csr_matrix, coo_matrix
import numpy as np
from src.models.gae.model import GCNModelVAE

class ClEncoder(nn.Module):
    def __init__(self, feature:torch.Tensor,device,dim_latent):
        super(ClEncoder, self).__init__()
        # self.fc1 = nn.Linear(feature.shape[1], feature.shape[1]).to(device)
        # self.fc1 = nn.Linear(feature.shape[1], 128*2)
        self.fc1 = nn.Linear(feature.shape[1], dim_latent*4).to(device)
        self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(feature.shape[1], feature.shape[1]).to(device)
        self.fc2 = nn.Linear(dim_latent*4, dim_latent).to(device)
        # self.fc2 = nn.Linear(128*2, 128)
        self.device = device
        # self.gae = gae_model.GCNModelVAE(feature.shape[1],feature.shape[1],feature.shape[1],0.1,self.device)
        self.gae = gae_model.GCNModelVAE(dim_latent,dim_latent,dim_latent,0.1,self.device)
        # self.gae = gae_model.GCNModelVAE(128,128,128,0.1)
        

    def forward(self, x,adj):
        x = x.to(self.device)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)

        # train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
        # data = np.ones(train_edges.shape[0])
        # adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
        # adj_train = adj_train + adj_train.T

        adj_norm = preprocess_graph(adj)
        adj_label = adj_train + sp.eye(adj_train.shape[0])
        adj_label = torch.FloatTensor(adj_label.toarray())

        z, mu, logvar = self.gae(x, adj_norm)
        # adj_label = adj_train + sp.eye(adj_train.shape[0])
        n_nodes, feat_dim = x.shape
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        #TODO gea拟合原样本损失，调整生成质量
        gae_loss = loss_function(self.device,preds=z, labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight)
        #TODO 图对比损失计算，调整mlp预编码质量
        print(f"gae损失{gae_loss}")

        return x[self.num_user:],gae_loss
        # return x

    # adj.getrow(0).nonzero()[1]
    def item_modal_forward(self,features,adj,num_user,d):
        index_query_feature_id = {}
        feature_id_query_index = {}
        row, col = self.get_neibor(adj, 0, 2)

        row_set, col_set = set(row), set(col)
        row_ls, col_ls = list(row_set), list(col_set)
        target_ind = list(row_ls) + list(col_ls)
        row_result = []
        col_result = []
        for i in range(len(target_ind)):
            index_query_feature_id[i] = target_ind[i]
            feature_id_query_index[target_ind[i]] = i
        for i, o in enumerate(row):
            row_result.append(feature_id_query_index[o])
        for i, o in enumerate(col):
            col_result.append(feature_id_query_index[o])
        new_coo_matrix = coo_matrix((np.array([1] * len(row_result)), (row_result, col_result)),
                                    shape=(len(target_ind), len(target_ind)))
        # dense_adj = new_coo_matrix.todense()
        # sub_graph_features = features[target_ind]

        #TODO 样本特征从item扩展到user
        self.num_user = num_user
        feature_users = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
        np.random.randn(num_user, features.shape[1]), dtype=torch.float32, requires_grad=True)))
        feature_users,features=feature_users.to(self.device),features.to(self.device)
        features = torch.cat((feature_users, features), dim=0)


        # features,adj = features.to(self.device),adj
        features,adj = features[target_ind].to(self.device),new_coo_matrix
        return self.forward(features,adj)

    # new_coo_matrix = coo_matrix((np.array([1] * len(a)), (a, b)), shape=(len(data), len(data)))
    # def cl_loss(self,feature):

    #TODO ！！！！！！！稀疏矩阵续写
    #     e
    # def neibor_iter(self,nodes_set:set,adj,n,current_order,k):
    #     if current_order > k:
    #         return nodes_set
    #     for i in adj.getrow(n).nonzero()[1]:
    #         nodes_set.add(i)
    #         self.neibor_iter(nodes_set,adj,i,current_order+1,k)
    #
    # def get_neibor(self,adj,n,k):
    #     return self.neibor_iter({},adj,n,0,k)

    def neibor_iter(self,row,col,adj,n,current_order,k,is_user:bool):
        if current_order < k:
            iter_ls = adj.getrow(n).nonzero()[1] if is_user else adj.getcol(n).nonzero()[0]
            # iter_ls = adj.getrow(n).nonzero()[1]
            for i in iter_ls:
                row.append(n if is_user else i)
                col.append(i if is_user else n)
                # print((i,not is_user,n))
                # nodes_set.add((i,not is_user))#TODO 三元组（结点id，是否是用户节点，根节点）
                # nodes_set.add((i,not is_user))#TODO 三元组（结点id，是否是用户节点，根节点）
                self.neibor_iter(row,col,adj,i,current_order+1,k,not is_user)

    def get_neibor(self,adj,n,k):
        # result = set([])
        # result = []
        row = []
        col = []
        # result.add((n,True))
        self.neibor_iter(row,col,adj,n,0,k,True)
        return row,col


