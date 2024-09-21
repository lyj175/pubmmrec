import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.gae.layers import GraphConvolution


class GCNModelVAE(nn.Module):#TODO 输入样本与邻接矩阵，输出VAE编码的新邻接矩阵与均值方差矩阵
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)#TODO 基础编码
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)#TODO 映射均值
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)#TODO 映射方差
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):#TODO 如果是训练阶段，返回重x的潜在表示z，否则返回x的均值
        if self.training:
            std = torch.exp(logvar)#TODO 方差还原为标准差
            eps = torch.randn_like(std)#TODO 从标准正态分布随机采样噪声
            return eps.mul(std).add_(mu)#TODO 对标准差位乘上噪声权重，达到增加噪声的效果，再加上均值矩阵输出
        else:
            return mu
    def forward(self, x, adj):#TODO 计算x的隐性表示z
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar,z


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))#TODO 节点转置相乘成对求内积，反映相关性，sigmoid归一化映射为0-1的概率表示
        return adj#TODO 返回最终生成的隐含z
