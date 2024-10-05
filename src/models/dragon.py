# coding: utf-8
#
# user-graph need to be generated by the following script
# tools/generate-u-u-matrix.py
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, degree
from src.models.metadata_split import *
from src.models.channel_attention_fusion import *
from src.models.gae import train
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('/content/drive/MyDrive/mmrec_revise_3/src/common'))))
from src.common.abstract_recommender import GeneralRecommender
from src.models import ClMLP
from scipy.sparse import coo_matrix
from src.models.vgae import VGAE

class DRAGON(GeneralRecommender):
    def __init__(self, config, dataset):
        super(DRAGON, self).__init__(config, dataset)

        num_user = self.n_users
        num_item = self.n_items
        batch_size = config['train_batch_size']  # not used
        dim_x = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config['mm_image_weight']
        has_id = True

        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.k = 40
        self.aggr_mode = config['aggr_mode']
        self.user_aggr_mode = 'softmax'
        self.num_layer = 1
        self.cold_start = 0
        self.dataset = dataset
        # self.construction = 'weighted_max'
        self.construction = 'cat'
        # self.construction = 'channel'
        self.reg_weight = config['reg_weight']
        self.drop_rate = 0.1
        self.v_rep = None
        self.t_rep = None
        self.v_preference = None
        self.t_preference = None
        self.dim_latent = 64
        self.dim_feat = 128
        self.MLP_v = nn.Linear(self.dim_latent, self.dim_latent, bias=False)
        self.MLP_t = nn.Linear(self.dim_latent, self.dim_latent, bias=False)
        self.mm_adj = None
        self.metadata_num = 10
        self.fusion_model = ChannelAttention(self.metadata_num)
        self.rep_for_fusion = None
        self.split_scale_index = None
        # self.clEncoder = ClMLP.ClEncoder(dim_x,dim_x,dim_x,self.device)


        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.user_graph_dict = np.load(os.path.join(dataset_path, config['user_graph_dict_file']),#TODO user-user的子图字典{user:[[交互的其他用户],[交互的用户的权重]}
                                       allow_pickle=True).item()

        mm_adj_file = os.path.join(dataset_path, 'mm_adj_{}.pt'.format(self.knn_k))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)#TODO 模态特征的线性变换层，输入特征数为图像的特征数，输出为64（特征变换后的大小，嵌入后）
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        else:
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)

        # packing interaction in training into edge_index
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)#TODO 交互边，稀疏矩阵方式存储（1,2）表示交互的item与user
        # train_interactions_for_gae = dataset.inter_matrix(form='coo').astype(np.float32)
        # train_interactions_for_gae.col += self.num_user#TODO 给item索引加上用户数，以便处理形成统一的邻接矩阵
        self.train_interactions = train_interactions
        edge_index = self.pack_edge_index(train_interactions)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)
        self.node_interaction_dict = np.load(os.path.join(dataset_path, 'node_interaction_dict.npy'),
                                       allow_pickle=True).item()

        #TODO 完整邻接矩阵user和item
        rows = edge_index[:, 0]
        rows = rows.tolist()
        cols = edge_index[:, 1]
        cols = cols.tolist()
        for node, (neighbors, weights) in self.user_graph_dict.items():
            rows+=[node]*len(neighbors)
            cols+=neighbors
            # for i, neighbor in enumerate(neighbors):
            #     rows = np.append(rows, node)
            #     cols = np.append(cols, neighbor)
        item_item_adj_path = os.path.join(dataset_path, 'image_adj_{}_{}.pt'.format(self.knn_k, True))
        item_item_adj = torch.load(item_item_adj_path).coalesce()
        i_i_rows = item_item_adj.indices()[0]+self.num_user
        i_i_cols = item_item_adj.indices()[1]+self.num_user
        rows+=i_i_rows
        cols+=i_i_cols

        self.coo_all_adj = coo_matrix(([1]*len(rows), (rows, cols)), shape=(self.num_user+self.num_item, self.num_user+self.num_item))

        # pdb.set_trace()
        self.weight_u = nn.Parameter(nn.init.xavier_normal_(#TODO 作用相当于对卷积后的特征再做一次线性变换
            torch.tensor(np.random.randn(self.num_user, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_u.data = F.softmax(self.weight_u, dim=1)#TODO 随机初始化用户权重数据[[[1],[1]],...]每条数据的权重两个特征表示

        self.weight_i = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_item, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_i.data = F.softmax(self.weight_i, dim=1)

        self.item_index = torch.zeros([self.num_item], dtype=torch.long)#TODO item索引数组
        index = []
        for i in range(self.num_item):
            self.item_index[i] = i
            index.append(i)
        self.drop_percent = self.drop_rate
        self.single_percent = 1
        self.double_percent = 0

        drop_item = torch.tensor(
            np.random.choice(self.item_index, int(self.num_item * self.drop_percent), replace=False))#TODO 随机选择百分之一的drop样本
        drop_item_single = drop_item[:int(self.single_percent * len(drop_item))]

        self.dropv_node_idx_single = drop_item_single[:int(len(drop_item_single) * 1 / 3)]#TODO drop样本数量分配三分之一给视觉样本，三分之一给文本样本
        self.dropt_node_idx_single = drop_item_single[int(len(drop_item_single) * 2 / 3):]

        self.dropv_node_idx = self.dropv_node_idx_single
        self.dropt_node_idx = self.dropt_node_idx_single

        mask_cnt = torch.zeros(self.num_item, dtype=int).tolist()
        for edge in edge_index:#TODO 把基于item索引的边加入到edge数组中
            mask_cnt[edge[1] - self.num_user] += 1#TODO 统计item边的条数，中item的标识是由user数为基础从1自增的吗？
        mask_dropv = []
        mask_dropt = []
        for idx, num in enumerate(mask_cnt):#TODO 掩去对应的drop样本的边，边索引处用True或False表示是否掩去
            temp_false = [False] * num
            temp_true = [True] * num
            mask_dropv.extend(temp_false) if idx in self.dropv_node_idx else mask_dropv.extend(temp_true)
            mask_dropt.extend(temp_false) if idx in self.dropt_node_idx else mask_dropt.extend(temp_true)
        #TODO T表示将矩阵行列互换，也就是将user和item转为两个数组，方便排序运算
        edge_index = edge_index[np.lexsort(edge_index.T[1, None])]#TODO 对边数据排序，先根据边的终点节点索引进行排序，如果终点节点索引相同，则再根据边的起点节点索引进行排序，方便邻居节点聚合运算
        edge_index_dropv = edge_index[mask_dropv]#TODO 布尔索引，留下对应索引位为True的元素
        edge_index_dropt = edge_index[mask_dropt]

        self.edge_index_dropv = torch.tensor(edge_index_dropv).t().contiguous().to(self.device)
        self.edge_index_dropt = torch.tensor(edge_index_dropt).t().contiguous().to(self.device)
        #TODO 给边加上方向，A-B变为A->B和B->A，方便卷积聚合运算
        self.edge_index_dropv = torch.cat((self.edge_index_dropv, self.edge_index_dropv[[1, 0]]), dim=1)
        self.edge_index_dropt = torch.cat((self.edge_index_dropt, self.edge_index_dropt[[1, 0]]), dim=1)
        #TODO 两个模态的特征输入（两倍特征），线性变换为单个样本的特征，模态融合后的下采样操作？
        self.MLP_user = nn.Linear(self.dim_latent * 2, self.dim_latent)

        if self.v_feat is not None:
            self.v_drop_ze = torch.zeros(len(self.dropv_node_idx), self.v_feat.size(1)).to(self.device)#TODO 把图像模态特征发送到gpu,节点数*嵌入维度(4090)
            self.v_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                             num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=64,
                             device=self.device, features=self.v_feat)  # 256)#TODO GCN图像模态特征编码
        if self.t_feat is not None:
            self.t_drop_ze = torch.zeros(len(self.dropt_node_idx), self.t_feat.size(1)).to(self.device)
            self.t_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                             num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=64,
                             device=self.device, features=self.t_feat)

        self.user_graph = User_Graph_sample(num_user, 'add', self.dim_latent)
        #TODO 定义模型中的可学习可调整参数，Xavier正态分布初始化，调整的就是所有用户与所有项目的最终嵌入向量
        self.result_embed = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)
        self.clVencoder = ClMLP.ClEncoder(self.v_feat, self.device, self.dim_latent, self.node_interaction_dict,self.num_user,self.num_item)
        self.clTencoder = ClMLP.ClEncoder(self.t_feat, self.device, self.dim_latent, self.node_interaction_dict,self.num_user,self.num_item)
        #TODO 生成对比梯度调整
        self.gae_v_optimizer = optim.Adam(self.clVencoder.parameters(), lr=0.01)
        self.gae_t_optimizer = optim.Adam(self.clVencoder.parameters(), lr=0.01)

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def pre_epoch_processing(self):
        self.epoch_user_graph, self.user_weight_matrix = self.topk_sample(self.k)
        self.user_weight_matrix = self.user_weight_matrix.to(self.device)#TODO 发送用户权重数据到计算单元

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        # ndarray([598918, 2]) for ml-imdb
        return np.column_stack((rows, cols))

    def test_gae_train(self):
        # parser = argparse.ArgumentParser()
        # parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
        # parser.add_argument('--seed', type=int, default=42, help='Random seed.')
        # parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
        # parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
        # parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
        # parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
        # parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
        # parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')
        #
        # args = parser.parse_args()
        # train.gae_for(args,self.t_feat,self.num_user,self.node_interaction_dict)
        # v = VGAE(self.num_user,self.num_item,self.node_interaction_dict,self.t_feat)
        # temp_r = v.simulate(2)
        # print(temp_r)
        pass

    #TODO forward就做了多模态特征融合、相似信息传播、推荐分数计算三个工作。其中初始的特征由GCN聚合编码得到
    def forward(self, interaction):#TODO 模型推理过程，interaction就是三个元素的二维数组，分别代表用户列表，以及对应用户的正负item列表
        print('forawrd_start')
        user_nodes, pos_item_nodes, neg_item_nodes = interaction[0], interaction[1], interaction[2]
        pos_item_nodes += self.n_users
        neg_item_nodes += self.n_users
        representation = None
        adj = self.dataset.get_adj_matrix()
        print('图像模态',self.v_feat.shape,'文本模态',self.t_feat.shape)
        if self.v_feat is not None:
            #TODO 输入交互边与图像模态，聚合传播存在交互的节点，v_rep：所有节点的图像表征（其中顺序为0-user_num为用户节点，往后为item节点），
            # v_preference：所有用户的图像特征偏好,self.v_rep 的维度通常是 (num_user + num_item, embedding_size)，其中 embedding_size 是特征的维度。
            # TODO 对比学习预编码
            # self.clVencoder.train()
            # self.gae_v_optimizer.zero_grad()
            # self.test_gae_train()#TODO 待删除
            # self.v_feat_,gae_loss_v = self.clVencoder.item_modal_forward(self.v_feat,adj,self.num_user,self.device)

            print('-----------开始VGAE-CL编码-图像模态------------')
            self.v_feat_,self.cl_loss_v,self.vgae_loss_v = self.clVencoder.item_modal_forward(self.v_feat)#TODO VGAE and Contrastive Learning 1 开始前向传播流程
            print('-----------结束VGAE-CL编码-图像模态------------')
            # gae_loss_v.backward()
            # self.gae_v_optimizer.step()

            # self.a,b = self.clVencoder.item_modal_forward(self.v_feat,adj,self.num_user,self.device)
            # self.v_rep, self.v_preference = self.v_gcn(self.edge_index_dropv, self.edge_index, self.v_feat)
            print("-----vgae_loss_v-------"+str(self.vgae_loss_v))
            #卷积传播
            self.v_rep, self.v_preference = self.v_gcn(self.edge_index_dropv, self.edge_index, self.v_feat_) #TODO 开启VGAE时
            representation = self.v_rep
        if self.t_feat is not None:
            # TODO 对比学习预编码
            # self.clTencoder = ClMLP.ClEncoder(self.t_feat, self.device,self.dim_latent)
            # self.clTencoder.train()wake
            # self.gae_t_optimizer.zero_grad()
            # self.t_feat_, gae_loss_t = self.clTencoder.item_modal_forward(self.t_feat, adj,self.num_user,self.device)
            # print('-----------开始VGAE-CL编码-文本模态------------')
            self.t_feat_, self.cl_loss_t,self.vgae_loss_t = self.clTencoder.item_modal_forward(self.t_feat)
            # print('-----------结束VGAE-CL编码-文本模态------------')
            # print("-----vgae_loss_t-------" + str(self.vgae_loss_t))
            # gae_loss_t.backward()
            # self.gae_t_optimizer.step()

            # self.b, gae_loss_t = self.clTencoder.item_modal_forward(self.t_feat, adj,self.num_user,self.device)
            # self.t_rep, self.t_preference = self.t_gcn(self.edge_index_dropt, self.edge_index, self.t_feat)
            self.t_rep, self.t_preference = self.t_gcn(self.edge_index_dropt, self.edge_index, self.t_feat_) #TODO 开启VGAE时
            if representation is None:
                representation = self.t_rep
            else:#TODO 模态融合1：先拼接所有节点的两个模态表示
                # if self.construction == 'cat':
                #     representation = torch.cat((self.v_rep, self.t_rep), dim=1)
                # else:
                #     representation += self.t_rep
                #TODO channel fusion
                if self.rep_for_fusion == None or self.split_scale_index == None:
                    self.rep_for_fusion,self.split_scale_index = metadata_split(self.metadata_num,self.v_rep)
                weight_fea = self.fusion_model(self.rep_for_fusion)
                weight_fea = weight_fea.squeeze()
                weight_fea = weight_fea.detach().numpy()
                for i in range(0,len(self.v_rep)):
                    split_fea_for_m_v = np.array_split(self.v_rep[i].detach().numpy(), self.split_scale_index[i])
                    m_result_v = torch.cat([torch.from_numpy(np.multiply(x, y)) for x, y in zip(split_fea_for_m_v, weight_fea[i])],0)
                    self.v_rep[i] = m_result_v

                    split_fea_for_m_t = np.array_split(self.t_rep[i].detach().numpy(), self.split_scale_index[i])
                    m_result_t = torch.cat(
                        [torch.from_numpy(np.multiply(x, y)) for x, y in zip(split_fea_for_m_t, weight_fea[i])], 0)
                    self.t_rep[i] = m_result_t
                if self.construction == 'cat':
                    representation = torch.cat((self.v_rep, self.t_rep), dim=1)
                else:
                    representation += self.t_rep

        # TODO 模态融合2：选择具体融合方式
        if self.construction == 'weighted_sum':
            if self.v_rep is not None:
                self.v_rep = torch.unsqueeze(self.v_rep, 2)#TODO 使用 torch.cat 将图像特征表示和文本特征表示沿着第三个维度拼接在一起，得到 (num_user + num_item, embedding_size, 2) 的矩阵。
                user_rep = self.v_rep[:self.num_user]
            if self.t_rep is not None:
                self.t_rep = torch.unsqueeze(self.t_rep, 2)
                user_rep = self.t_rep[:self.num_user]
            if self.v_rep is not None and self.t_rep is not None:
                #TODO 拼接后的模态特征矩阵和用户权重矩阵(三维：用户、图像权重、文本权重)相乘，得到最终的user多模态表示
                user_rep = torch.matmul(torch.cat((self.v_rep[:self.num_user], self.t_rep[:self.num_user]), dim=2),
                                        self.weight_u)
            user_rep = torch.squeeze(user_rep)#TODO 去除维度为1的维度，保留最终的用户二维表示

        if self.construction == 'weighted_max':
            # pdb.set_trace()
            self.v_rep = torch.unsqueeze(self.v_rep, 2)

            self.t_rep = torch.unsqueeze(self.t_rep, 2)

            user_rep = torch.cat((self.v_rep[:self.num_user], self.t_rep[:self.num_user]), dim=2)
            user_rep = self.weight_u.transpose(1, 2) * user_rep
            user_rep = torch.max(user_rep, dim=2).values
        if self.construction == 'cat':
            # pdb.set_trace()
            if self.v_rep is not None:
                user_rep = self.v_rep[:self.num_user]
            if self.t_rep is not None:
                user_rep = self.t_rep[:self.num_user]
            if self.v_rep is not None and self.t_rep is not None:
                self.v_rep = torch.unsqueeze(self.v_rep, 2)
                self.t_rep = torch.unsqueeze(self.t_rep, 2)
                user_rep = torch.cat((self.v_rep[:self.num_user], self.t_rep[:self.num_user]), dim=2)
                user_rep = self.weight_u.transpose(1, 2) * user_rep

                user_rep = torch.cat((user_rep[:, :, 0], user_rep[:, :, 1]), dim=1)


        item_rep = representation[self.num_user:]
        #TODO 物品多模态信息融合
        ############################################ multi-modal information aggregation
        h = item_rep
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)#TODO 物品相似度矩阵，乘上用户表示矩阵，通过相似度聚合物品表示，
            # 假如self.mm_adj =
            # [[1.0, 0.5, 0.2, 0.1, 0.0],显然类似一个残差链接，当前物品本身的特征加上与其他物品相似度权重求和特征
              # [0.5, 1.0, 0.3, 0.2, 0.1],
              # [0.2, 0.3, 1.0, 0.4, 0.2],
              # [0.1, 0.2, 0.4, 1.0, 0.3],
              # [0.0, 0.1, 0.2, 0.3, 1.0]]
        h_u1 = self.user_graph(user_rep, self.epoch_user_graph, self.user_weight_matrix)#TODO 同样的用户相似度聚合
        user_rep = user_rep + h_u1#TODO 论文图示！原特征与聚合特征的残差链接
        item_rep = item_rep + h
        # self.result_embed = torch.cat((user_rep, item_rep), dim=0)#TODO ！！大幅降低模型性能。拼接得到最终的users+items的表示
        self.result_embed = nn.Parameter(torch.cat((user_rep, item_rep), dim=0))#TODO 拼接得到最终的users+items的表示
        user_tensor = self.result_embed[user_nodes]
        pos_item_tensor = self.result_embed[pos_item_nodes]
        neg_item_tensor = self.result_embed[neg_item_nodes]
        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)#TODO 内积，按行求和
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)
        # return pos_scores, neg_scores
        print('forawrd_end')
        return pos_scores, neg_scores

    # TODO 对比学习损失：计算正例分数与负例分数和之比的负对数 弃用了
    def InfoNCE(self, view1, view2, temp):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temp)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temp).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    #TODO 每个批次训练时计算损失
    def  calculate_loss(self, interaction):
        user = interaction[0]
        pos_scores, neg_scores = self.forward(interaction)#TODO 前向传播
        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        reg_embedding_loss_v = (self.v_preference[user] ** 2).mean() if self.v_preference is not None else 0.0#TODO L2正则损失防止参数过大过拟合，参数平方均值
        reg_embedding_loss_t = (self.t_preference[user] ** 2).mean() if self.t_preference is not None else 0.0

        reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_t)
        if self.construction == 'weighted_sum':
            reg_loss += self.reg_weight * (self.weight_u ** 2).mean()
            reg_loss += self.reg_weight * (self.weight_i ** 2).mean()
        elif self.construction == 'cat':
            reg_loss += self.reg_weight * (self.weight_u ** 2).mean()
        elif self.construction == 'cat_mlp':
            reg_loss += self.reg_weight * (self.MLP_user.weight ** 2).mean()

        # cl_loss = self.InfoNCE(self.v_preference,self.t_preference,1)

        # return loss_value + reg_loss
        loss_count = loss_value + reg_loss + self.cl_loss_v
        # loss_count = loss_value + reg_loss + self.cl_loss_v + self.cl_loss_t + self.vgae_loss_t + self.vgae_loss_v
        print(f'总损失{loss_count}')
        #TODO 最终的损失：基础top-k排序损失+正则损失+对比损失
        return loss_count

        # return loss_value + reg_loss

    def full_sort_predict(self, interaction):
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

    def topk_sample(self, k):
        user_graph_index = []
        count_num = 0
        user_weight_matrix = torch.zeros(len(self.user_graph_dict), k)
        tasike = []
        for i in range(k):
            tasike.append(0)
        for i in range(len(self.user_graph_dict)):
            if len(self.user_graph_dict[i][0]) < k:
                count_num += 1
                if len(self.user_graph_dict[i][0]) == 0:
                    # pdb.set_trace()
                    user_graph_index.append(tasike)
                    continue
                user_graph_sample = self.user_graph_dict[i][0][:k]
                user_graph_weight = self.user_graph_dict[i][1][:k]
                while len(user_graph_sample) < k:
                    rand_index = np.random.randint(0, len(user_graph_sample))
                    user_graph_sample.append(user_graph_sample[rand_index])
                    user_graph_weight.append(user_graph_weight[rand_index])
                user_graph_index.append(user_graph_sample)

                if self.user_aggr_mode == 'softmax':
                    user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
                if self.user_aggr_mode == 'mean':
                    user_weight_matrix[i] = torch.ones(k) / k  # mean
                continue
            user_graph_sample = self.user_graph_dict[i][0][:k]
            user_graph_weight = self.user_graph_dict[i][1][:k]

            if self.user_aggr_mode == 'softmax':
                user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
            if self.user_aggr_mode == 'mean':
                user_weight_matrix[i] = torch.ones(k) / k  # mean
            user_graph_index.append(user_graph_sample)

        # pdb.set_trace()
        return user_graph_index, user_weight_matrix

#TODO 聚合存在交互的用户
class User_Graph_sample(torch.nn.Module):
    def __init__(self, num_user, aggr_mode, dim_latent):
        super(User_Graph_sample, self).__init__()
        self.num_user = num_user
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode

    def forward(self, features, user_graph, user_matrix):
        index = user_graph
        u_features = features[index]#TODO 与index用户交互过的其他用户的索引
        user_matrix = user_matrix.unsqueeze(1)#TODO 用户权重矩阵
        # pdb.set_trace()
        u_pre = torch.matmul(user_matrix, u_features)#TODO 加权求和
        u_pre = u_pre.squeeze()
        return u_pre


class GCN(torch.nn.Module):
    def __init__(self, datasets, batch_size, num_user, num_item, dim_id, aggr_mode, num_layer, has_id, dropout,
                 dim_latent=None, device=None, features=None):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.datasets = datasets
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.num_layer = num_layer
        self.has_id = has_id
        self.dropout = dropout
        self.device = device

        if self.dim_latent:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(#TODO 用户的最终表示就是需要调整的参数
                np.random.randn(num_user, self.dim_latent), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device))
            self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
            self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)#TODO 真正图卷积调用的类

        else:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(num_user, self.dim_feat), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device))
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)
    #TODO GCN对特征开始卷积聚合
    def forward(self, edge_index_drop, edge_index, features):
        # temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features#TODO 先使用模型定义的线性层进行线性变换降维，再用模型定义的MLP提取特征
        # x = torch.cat((self.preference, temp_features), dim=0).to(self.device)#TODO 前user_num行都是表示用户的向量（传播前随机初始化），后面都是item的向量
        print(features.shape,self.preference.shape)

        x = F.normalize(features).to(self.device)
        h = self.conv_embed_1(x, edge_index)#TODO 两层卷积  # equation 1
        h_1 = self.conv_embed_1(h, edge_index)

        x_hat = h + x + h_1
        return x_hat, self.preference

class Base_gcn(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        # pdb.set_trace()
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
            # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        # pdb.set_trace()
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)#TODO gcn传播的落点

    def message(self, x_j, edge_index, size):
        if self.aggr == 'add':
            # pdb.set_trace()
            row, col = edge_index#TODO 起点和终点边的索引
            deg = degree(row, size[0], dtype=x_j.dtype)#TODO 每个节点的度
            deg_inv_sqrt = deg.pow(-0.5)#TODO 通过度取其倒数的平方根，度高代表流行，对偏好贡献就少
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]#TODO 边的两个节点度权重相乘得到边的权重
            return norm.view(-1, 1) * x_j#TODO 归一化的边权重乘上对应的边,得到最终边的表示
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


