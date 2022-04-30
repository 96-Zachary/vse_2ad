import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class AP(nn.Module):
    def __init__(self, d_hidden):
        super(AP, self).__init__()
        self.linear = nn.Linear(d_hidden, d_hidden)
        self.balance = nn.Linear(d_hidden, 1)
        self.relu = nn.ReLU(inplace=True)

        self.init_weights()

    def init_weights(self):
        r = np.sqrt(6.) / np.sqrt(self.linear.in_features +
                                  self.linear.out_features)
        self.linear.weight.data.uniform_(-r, r)
        self.linear.bias.data.fill_(0)

    def forward(self, features, lengths):
        """
        :param features: features with shape B x K x D
        :param lengths: B x 1, specify the length of each data sample.
        :return: pooled feature with shape B x D
        """
        max_len = features.size(1)
        mask = torch.arange(max_len).expand(features.size(0), features.size(1)).to(lengths.device)
        mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1)
        mask_features = features.masked_fill(mask == 0, -1000)
        mask_features = mask_features.sort(dim=1, descending=True)[0]

        # embedding-level
        embed_weights = F.softmax(mask_features, dim=1)
        embed_features = (mask_features * embed_weights).sum(1)

        # token-level
        # token_weights = [B x K x D]
        mask_features = mask_features.masked_fill(mask == 0, 0)
        token_weights = self.linear(mask_features)
        token_weights = F.softmax(self.relu(token_weights),
                                  dim=1)
        token_features = (mask_features * token_weights).sum(dim=1)
        fusion_features = torch.cat([token_features.unsqueeze(1),
                                     embed_features.unsqueeze(1)],
                                    dim=1)
        fusion_weights = F.softmax(self.balance(fusion_features),
                                   dim=1)
        pool_features = (fusion_features * fusion_weights).sum(1)

        return pool_features, fusion_weights.squeeze()