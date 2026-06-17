"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch #é uma biblioteca de código aberto focada em aprendizado de máquina e deep learning
import torch.nn as nn

L = 10 # labels

def jaccard_similarity(i, j):
    """Computes the Jaccard similarity between two lists of length L."""

    sum_min = 0
    sum_max = 0

    for l in range(L):
        sum_min = sum_min + min(i[l], j[l])
        sum_max = sum_max + max(i[l], j[l])

    if sum_max == 0:
        return 0
    return sum_min / sum_max

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0] # N
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device) # This command creates a 2D identity matrix of size batch_size × batch_size, sets its 
                                                                        # numerical precision to 32-bit float, and loads it onto a specific computation device (e.g., CPU or GPU).

        elif labels is not None:
            labels = labels.contiguous().view(-1, 1) # is used to flatten a tensor of target labels into a 2D column vector with a single column. 
                                                    # It is commonly used in binary classification, regression, or loss functions (like BCELoss) to resolve shape mismatch errors.
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device) #  creates a binary similarity matrix that identifies which samples in a batch share the exact same label, 
                                                                # a foundational step in contrastive learning losses like SupCon
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # rearrange multi-view or batched features
                                                                        # It flattens the first two dimensions of a tensor, transforming data with multiple "views" per sample into a single sequential list of samples.
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div( # divide todas as similaridades pela temperatura
            torch.matmul(anchor_feature, contrast_feature.T), # matmul realiza uma multiplicação de matrizes 
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) # varre a matriz anchor_dot_contrast e retorna o maior valor de similaridade 
                                                    # keepdim: mantém a dimensão original, uma coluna vertical
                # _ : ignora os índices dos valores máximos, pois não são necessários para o cálculo da perda.
        logits = anchor_dot_contrast - logits_max.detach() # ----------------------------------------------------------------

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class KPositiveContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, k_pos=6):
        super(KPositiveContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.k_pos = k_pos

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: Tensor of shape [batch_size, embedding_dim]
            labels: LongTensor of shape [batch_size]
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]
        
        # 1. Normalize embeddings to unit hypersphere
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 2. Compute cosine similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.t()) # [B, B]
        
        # Create identity mask (to exclude the anchor itself)
        logits_mask = torch.scatter(
            torch.ones_like(similarity_matrix),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        
        loss_total = 0.0
        valid_anchors = 0
        
        for i in range(batch_size):
            anchor_label = labels[i].item()
            
            # 3. Identify all positive instances in the batch for the current anchor
            # (excluding the anchor itself)
            pos_indices = (labels == anchor_label).nonzero(as_tuple=True)[0]
            pos_indices = pos_indices[pos_indices != i]
            
            if len(pos_indices) == 0:
                continue # Skip if there are no other positives in this batch
                
            # 4. If we have more than k positives, sample exactly k of them
            if len(pos_indices) > self.k_pos:
                perm = torch.randperm(len(pos_indices))
                pos_indices = pos_indices[perm[:self.k_pos]]
                
            # 5. Extract the logits (similarities) for these k positives
            pos_logits = similarity_matrix[i, pos_indices] / self.temperature # [K]
            
            # 6. Extract negative logits (all samples excluding the anchor and the selected positives)
            # Mask out the anchor itself, and the chosen k positive pairs
            current_mask = logits_mask[i].clone()
            current_mask[pos_indices] = 0
            neg_indices = current_mask.nonzero(as_tuple=True)[0]
            
            if len(neg_indices) == 0:
                continue # Skip if there are no negatives to push away
                
            neg_logits = similarity_matrix[i, neg_indices] / self.temperature # [N_neg]
            
            # 7. Concatenate pos logits and neg logits to calculate CrossEntropy
            # Shape: [1 + K + N_neg] -> target is always the 0-th index
            logits = torch.cat([pos_logits, neg_logits], dim=0).unsqueeze(0)
            target = torch.zeros(1, dtype=torch.long).to(device)
            
            # 8. Compute InfoNCE Loss for this specific anchor
            loss = F.cross_entropy(logits, target)
            loss_total += loss
            valid_anchors += 1
            
        if valid_anchors == 0:
            return torch.tensor(0.0, requires_grad=True).to(device)
            
        return loss_total / valid_anchors
