import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
import torch_geometric

# ... (Your existing code for DRAGON, GCN, Base_gcn, etc.)

class ContrastiveLearning(nn.Module):
    def __init__(self, dim_latent, temperature=0.1):
        super(ContrastiveLearning, self).__init__()
        self.dim_latent = dim_latent
        self.temperature = temperature

    def forward(self, user_rep, item_rep, edge_index):
        """
        对比学习模块，计算用户/物品正负例的对比损失

        Args:
            user_rep: 用户特征表示 (num_users, dim_latent)
            item_rep: 物品特征表示 (num_items, dim_latent)
            edge_index: 用户-物品交互关系的边信息 (2, num_edges)

        Returns:
            contrastive_loss: 对比学习损失
        """
        # 构建正负例
        user_nodes, pos_item_nodes = edge_index[0], edge_index[1]
        neg_item_nodes = self.sample_negative_items(user_nodes, pos_item_nodes, item_rep.size(0))

        # 计算相似度
        user_tensor = user_rep[user_nodes]
        pos_item_tensor = item_rep[pos_item_nodes]
        neg_item_tensor = item_rep[neg_item_nodes]

        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1) / self.temperature
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1) / self.temperature

        # 计算对比损失
        contrastive_loss = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))

        return contrastive_loss

    def sample_negative_items(self, user_nodes, pos_item_nodes, num_items):
        """
        随机采样负例物品
        """
        neg_item_nodes = torch.randint(0, num_items, (len(user_nodes),))
        # 确保负例物品不与正例物品相同
        while torch.any(neg_item_nodes == pos_item_nodes):
            neg_item_nodes = torch.randint(0, num_items, (len(user_nodes),))
        return neg_item_nodes