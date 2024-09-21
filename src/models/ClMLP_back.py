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
import torch.nn.functional as F
from src.models.gae.model import GCNModelVAE
from queue import Queue

class ClEncoder(nn.Module):
    def __init__(self, feature:torch.Tensor,device,dim_latent,node_interaction_dict):
        super(ClEncoder, self).__init__()
        # self.fc1 = nn.Linear(feature.shape[1], feature.shape[1]).to(device)
        # self.fc1 = nn.Linear(feature.shape[1], 128*2)
        self.node_interaction_dict =node_interaction_dict
        self.fc1 = nn.Linear(feature.shape[1], dim_latent*4).to(device)
        self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(feature.shape[1], feature.shape[1]).to(device)
        self.fc2 = nn.Linear(dim_latent*4, dim_latent).to(device)
        # self.fc2 = nn.Linear(128*2, 128)

        # self.MLP = nn.Linear(feature.shape[1], 4 * dim_latent)
        # self.MLP_1 = nn.Linear(4 * dim_latent, dim_latent)
        # self.relu = F.leaky_relu()
        # temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features#TODO 先使用模型定义的线性层进行线性变换降维，再用模型定义的MLP提取特征
        # x = torch.cat((self.preference, temp_features), dim=0).to(self.device)#TODO 前user_num行都是表示用户的向量（传播前随机初始化），后面都是item的向量

        self.device = device
        # self.gae = gae_model.GCNModelVAE(feature.shape[1],feature.shape[1],feature.shape[1],0.1,self.device)
        # self.gae = gae_model.GCNModelVAE(dim_latent,dim_latent,dim_latent,0.1,self.device)
        self.gae = gae_model.GCNModelVAE(dim_latent,dim_latent,dim_latent,0.1)
        # self.gae = gae_model.GCNModelVAE(128,128,128,0.1)
        

    def forward(self, feature_sub_graph,adj,ori_adj,ori_feature):
        feature = ori_feature.to(self.device)
        x = self.fc1(feature)
        x = self.relu(x)
        x = self.fc2(x)

        x_1 = self.fc1(feature_sub_graph)
        x_1 = self.relu(x_1)
        x_1 = self.fc2(x_1)

        # x = F.leaky_relu(self.MLP(x))
        # x = self.MLP_1(x)


        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)

        # train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
        # data = np.ones(train_edges.shape[0])
        # adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
        # adj_train = adj_train + adj_train.T

        adj_norm = preprocess_graph(adj)
        adj_label = adj_train + sp.eye(adj_train.shape[0])
        adj_label = torch.FloatTensor(adj_label.toarray())

        z, mu, logvar = self.gae(x_1, adj_norm)
        # adj_label = adj_train + sp.eye(adj_train.shape[0])
        n_nodes, feat_dim = x_1.shape
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        #TODO gea拟合原样本损失，调整生成质量
        gae_loss = loss_function(self.device,preds=z, labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight)
        #TODO 图对比损失计算，调整mlp预编码质量
        print(f"gae损失{gae_loss}")


        # return x,0
        return x,gae_loss
        # return x[self.num_user:],gae_loss
        # return x

    # adj.getrow(0).nonzero()[1]
    def item_modal_forward(self,features,adj:coo_matrix,num_user,d):
        # TODO 样本特征从item扩展到user
        self.num_user = num_user
        feature_users = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
            np.random.randn(num_user, features.shape[1]), dtype=torch.float32, requires_grad=True)))
        # feature_users, features = feature_users.to(self.device), features.to(self.device)
        self.features_new = torch.cat((feature_users, features), dim=0).to(self.device)

        all_count = []
        subgraph_data_row = []
        subgraph_data_col = []
        root_node_index = 20000

        # root = self.node_interaction_dict[root_node_index].tolist()
        subgraph_data_row = []
        subgraph_data_col = []
        all_count = set()
        # self.iter_node(root_node_index,subgraph_data_row,subgraph_data_col,1000,all_count)
        subgraph_data_row,subgraph_data_col,all_count = self.iter_node(0,1000)

        # subgraph_data_row+=len(root)*[root_node_index]
        # subgraph_data_col+=root
        # all_count+=len(root)*[root_node_index]
        # all_count+=root
        # for i in root:
        #     subgraph_data_row += len(self.node_interaction_dict[i]) * [i]
        #     subgraph_data_col += self.node_interaction_dict[i].tolist()
        #     all_count += len(self.node_interaction_dict[i]) * [i]
        #     all_count += self.node_interaction_dict[i].tolist()

        # all_count = set(all_count)
        index_query_feature_id,feature_id_query_index,features_subgraph = self.id_map(all_count)
        features_subgraph = torch.stack(features_subgraph)
        print(f'去重前{len(subgraph_data_row)}去重后{len(set(all_count))}')

        subgraph_data_row,subgraph_data_col = self.get_norm_index(subgraph_data_row,subgraph_data_col,feature_id_query_index)
        new_coo_matrix = coo_matrix((np.array([1] * len(subgraph_data_row)), (subgraph_data_row, subgraph_data_col)),
                                    shape=(len(all_count), len(all_count)))
        features_sub_graph,new_adj = features_subgraph,new_coo_matrix
        return self.forward(features_sub_graph,new_adj,adj,features)

    def iter_node(self,root_index,until):
        node_queue = Queue()
        node_queue.put(root_index)
        rows,cols,keys = [],[],[]
        while node_queue.qsize()>0:
            cur = node_queue.get()
            children = self.node_interaction_dict[cur].tolist()[:10]#TODO 只取前十个交互结点
            rows+=[cur]*len(children)
            cols+=children
            keys = list(keys)
            keys +=[cur]*len(children)
            keys +=children
            keys = set(keys)

            if len(keys) >= until:return rows,cols,keys
            for i in children:
                node_queue.put(i)

    def id_map(self,data):
        index_query_feature_id = {}
        feature_id_query_index = {}
        features = []
        data = list(data)
        for i in range(0,len(data)):
            index_query_feature_id[i] = data[i]
            feature_id_query_index[data[i]] = i
            features.append(self.features_new[i].clone())
        return index_query_feature_id,feature_id_query_index,features

    def get_norm_index(self,row,col,id_to_index_dict):
        new_row,new_col = [],[]
        for i in range(0,len(row)):
            new_row.append(id_to_index_dict[row[i]])
            new_col.append(id_to_index_dict[col[i]])
        return new_row,new_col

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


