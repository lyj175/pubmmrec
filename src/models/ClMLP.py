import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.vgae import VGAE
from models.gae import model as gae_model
from models.gae.utils import load_data, mask_test_edges, preprocess_graph, get_roc_score
import scipy.sparse as sp
from models.gae.optimizer import loss_function
from scipy.sparse import csr_matrix, coo_matrix
import numpy as np
import torch.nn.functional as F
from models.gae.model import GCNModelVAE
from queue import Queue
from sklearn.metrics.pairwise import cosine_similarity

class ClEncoder(nn.Module):
    def __init__(self, feature:torch.Tensor,device,dim_latent,node_interaction_dict,num_user,num_item):
        super(ClEncoder, self).__init__()
        # self.fc1 = nn.Linear(feature.shape[1], feature.shape[1]).to(device)
        # self.fc1 = nn.Linear(feature.shape[1], 128*2)
        self.node_interaction_dict =node_interaction_dict
        self.fc1 = nn.Linear(feature.shape[1], dim_latent*4).to(device)
        self.relu = nn.ReLU()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_latent = dim_latent
        # self.fc2 = nn.Linear(feature.shape[1], feature.shape[1]).to(device)
        self.fc2 = nn.Linear(dim_latent*4, dim_latent).to(device)
        self.fc3 = nn.Linear(dim_latent, dim_latent).to(device)
        self.vgae = VGAE(self.num_user, self.num_item, self.node_interaction_dict, feature,dim_latent)

        self.device = device
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        # self.gae = gae_model.GCNModelVAE(dim_latent,dim_latent,dim_latent,0.1)

    def forward(self, features):
        feature = features.to(self.device)
        # x = self.fc1(feature)
        # x = self.relu(x)
        # x = self.fc2(x)
        # TODO VGAE and Contrastive Learning 3 VGAE模块的MLP部分，对原始数据进行基础编码
        x_1 = self.fc1(feature)
        x_1 = self.relu(x_1)
        x_1 = self.fc2(x_1)
        x_1 = self.relu(x_1)
        # 加入初始化的user数据
        feature_users = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
            np.random.randn(self.num_user, self.dim_latent), dtype=torch.float32, requires_grad=True)))
        # feature_users = feature_users.to(self.device)
        #print('----------------',self.device)
        feature_users = feature_users.to(self.device)
        x_1 = x_1.to(self.device)
        x_1 = torch.cat((feature_users, x_1), dim=0)
        # x_1 = torch.cat((feature_users, x_1), dim=0).to(self.device)
        x_1 = self.fc3(x_1)

        self.vgae.set_fea(x_1)
        # TODO VGAE and Contrastive Learning 4 随机抽取子图并由VGAE生成对应增强子图
        results,vgae_loss = self.vgae.simulate(90)
        # cl_loss = 0
        cl_loss = self.info_nce_loss(results)
        # self.optimizer.zero_grad()
        # cl_loss.backward(retain_graph=True) #TODO 即时
        # self.optimizer.step()
        # if cl_loss > 1e6 or cl_loss < -1e6:
        #     print("Tensor may be exploding!")
        #     cl_loss = 0
        print('对比损失',cl_loss)
        # return x,cl_loss
        # TODO VGAE and Contrastive Learning 5 返回经过对比学习优化后的输出x_1与对比损失
        return x_1,cl_loss,vgae_loss

    def info_nce_loss(self,data,temperature=100):
        pos = torch.tensor(0.0)
        neg = torch.tensor(0.0)
        for i in range(0, len(data[0])):
            # print(i)
            # pos = pos + torch.log(torch.sum(torch.mm(data[0][i], data[0][i].T)))
            # pos = pos + torch.log(torch.sum(torch.mm(data[0][i], data[1][i].T))) #enhanced

            pos = pos + np.sum(cosine_similarity(data[0][i].cpu().detach().numpy(), data[0][i].cpu().detach().numpy()))
            pos = pos + np.sum(cosine_similarity(data[0][i].cpu().detach().numpy(), data[1][i].cpu().detach().numpy()))

            # pos = pos + np.sum(cosine_similarity(data[0][i].detach().numpy(), data[0][i].detach().numpy()))
            # pos = pos + np.sum(cosine_similarity(data[0][i].detach().numpy(), data[1][i].detach().numpy()))

            # pos += sum(torch.mm(data[0][i], data[0][i].T))
            # pos += sum(torch.mm(data[0][i], data[1][i].T))
            for j in range(0, len(data[0])):
                if j == i: continue
                # neg += torch.log(torch.sum(torch.mm(data[0][i], data[0][j].T)))
                # neg += torch.log(torch.sum(torch.mm(data[0][i], data[1][j].T)))

                neg = neg + np.sum(cosine_similarity(data[0][i].cpu().detach().numpy(), data[0][j].cpu().detach().numpy()))
                neg = neg + np.sum(cosine_similarity(data[0][i].cpu().detach().numpy(), data[1][j].cpu().detach().numpy()))
        # pos = pos / temperature
        # neg = neg / temperature
        # loss = -torch.log(torch.exp(pos) / (torch.exp(pos) + torch.exp(neg)))
        # loss = -torch.log(torch.exp(pos) / (torch.exp(pos) + torch.exp(neg)))
        # return torch.mean(loss)
        print('-----正例分数',pos,'-----反例分数',neg)
        return neg/pos

        # data[0]原始图的集合,data[1]生成图的数量
        # ori_,gen_ = data[0],data[1]
        # a = torch.mm(data[0][0], data[0][0].t())
        # # import pdb
        # # pdb.set_trace()
        # positive_score = torch.tensor(0.0)
        # negative_score = torch.tensor(0.0)
        # for i in range(0,len(ori_)):
        #     #正例对
        #     positive_score += torch.cosine_similarity(ori_[i],gen_[i],dim=1).sum()
        #     negative_score += self.all_neg_score(i,data)
        # return -torch.log((torch.exp(positive_score / temperature) / negative_score/temperature)).mean()
        # return 0


    def all_neg_score(self,target_index,data):
        linked_data = data[0][:target_index]+data[0][target_index:]+data[1][:target_index]+data[1][target_index:]#所有非目标
        # temp_data_compress = torch.sum(data[0][target_index],dim=0)
        loss_count = torch.tensor(0.0)
        for i in linked_data:
            # loss_count += torch.cosine_similarity(temp_data_compress, torch.sum(i,dim=0), dim=0)
            loss_count += torch.cosine_similarity(data[0][target_index], i, dim=1).sum()
        return loss_count


    # adj.getrow(0).nonzero()[1]
    def item_modal_forward(self,features):
        # torch.save(features, 'my_tensor.pt')
        # features.save()
        # TODO VGAE and Contrastive Learning 2 用户偏好向量生成
        # TODO 样本特征从item扩展到user,由于一开始没有用户特征（即用户偏好表示），所以随机初始化
        # feature_users = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
        #     np.random.randn(self.num_user, features.shape[1]), dtype=torch.float32, requires_grad=True)))
        # # feature_users, features = feature_users.to(self.device), features.to(self.device)
        # # TODO 将随机的用户特征和物品的特征进行拼接，得到数据集矩阵，0-num_user个向量是用户向量，剩下的是物品信息向量
        # features_new = torch.cat((feature_users, features), dim=0).to(self.device)

        # return self.forward(features_new)
        return self.forward(features)

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


