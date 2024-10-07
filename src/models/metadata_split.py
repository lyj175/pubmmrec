import numpy as np
import torch


def metadata_split(split_num,features):
    # split_num必须大于3
    print('生成metadata')
    mu = 0
    sigma = 1
    s = np.random.normal(mu, sigma, split_num)
    s = np.square(s)
    sum_of_values = np.sum(s)
    for i in range(0, len(s)-1):
        s[i] = s[i] / sum_of_values
    s = s*len(features[0])
    meta_data_len = int(np.max(s))

    # temp_fea = features.clone().detach()
    temp_fea = features
    # 供通道注意力计算的张量（样本数，metadata数量，metadata高，metadata宽）
    v_for_channel = torch.zeros(len(temp_fea), split_num, 1, meta_data_len)
    all_of_index = []#记录每个样本的元数据分割索引
    for j in range(0,len(temp_fea)):
        index = 0
        indexes = []
        metadatas = torch.zeros(split_num, 1, meta_data_len)#一个样本的所有metadata
        splitted_fea = temp_fea[j][index:index+int(s[0])]
        index = index+int(s[0])
        indexes.append(index)
        metadatas[0] = (join_tail(splitted_fea,meta_data_len))
        for i in range(1,split_num-1):
            splitted_fea = temp_fea[j][index:index + int(s[i])]
            index = index + int(s[i])
            indexes.append(index)
            metadatas[i] = (join_tail(splitted_fea,meta_data_len))#一个metadata
        metadatas[split_num-1] = (join_tail(temp_fea[j][index:index + int(s[split_num-1])], meta_data_len))
        v_for_channel[j] = metadatas
        all_of_index.append(indexes)
    print('生成metadata-finished')
    return v_for_channel,all_of_index

def join_tail(fea,len):
    fea = fea.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return torch.cat((fea, torch.zeros(len - fea.shape[0]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))), 0)