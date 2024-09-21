import torch
import torch.nn.modules.loss
import torch.nn.functional as F
import torch.nn as nn


def loss_function(device,preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    # cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=torch.tensor(pos_weight))
    # cost = norm * F.binary_cross_entropy_with_logits(preds, labels.to_dense(), pos_weight=torch.tensor(pos_weight))
    # cost = norm * F.binary_cross_entropy(torch.sigmoid(preds), labels, pos_weight=torch.tensor(pos_weight))
    
    preds = preds.to(device)
    labels = labels.to(device)
    # loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
    # loss_fn = nn.BCELoss(pos_weight=torch.tensor(pos_weight).to(device))
    # cost = loss_fn(torch.sigmoid(preds), labels)
    loss_fn = nn.BCELoss(reduction='none')  # No reduction for weighted sum
    cost = (loss_fn(torch.sigmoid(preds.to_dense()), labels.to_dense()) * labels * pos_weight).sum() / labels.sum()

    # cost = loss_fn(preds, labels)


    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD
