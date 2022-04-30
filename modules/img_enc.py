import numpy as np

import torch
import torch.nn as nn

from utils import l2norm
from modules.adcap import AP
from modules.mlp import MLP
from transformers import ViTModel


# Vision Encoder
class EncoderImage_ProjFeature(nn.Module):
    def __init__(self, img_dim, emb_size, precomp_enc_type='basic', no_imgnorm=False):
        super(EncoderImage_ProjFeature, self).__init__()
        self.emb_size = emb_size
        self.img_dim = img_dim
        self.precomp_enc_type = precomp_enc_type
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(self.img_dim, self.emb_size)
        self.mlp = MLP(img_dim, emb_size // 2, emb_size, 2)
        self.pool = AP(self.emb_size)

        self.init_weights()

    def init_weights(self):
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, img, img_lengths):
        '''
        Extract image features:
        input: img = [batch_size, n_region, img_dim]
        output: features = [batch_size, n_region, emb_size]
        '''
        img_emb = self.fc(img)

        if self.precomp_enc_type == 'basic':
            # features = [batch_size, n_region, emb_size]
            img_emb = self.mlp(img) + img_emb

        # pooling operation
        img_lengths = torch.LongTensor(img_lengths).to(img_emb.device)
        img_emb, _ = self.pool(img_emb, img_lengths)

        # normalization
        if not self.no_imgnorm:
            img_emb = l2norm(img_emb, dim=-1)

        return img_emb


class EncoderImage_BUTD(nn.Module):
    def __init__(self, backbone_cnn, img_dim, emb_size,
                 precomp_enc_type='basic', no_imgnorm=False):
        super(EncoderImage_BUTD, self).__init__()
        self.backbone = backbone_cnn
        self.img_proj = EncoderImage_ProjFeature(img_dim, emb_size,
                                                 precomp_enc_type, no_imgnorm)

    def forward(self, img, img_lengths=None):
        base_features = self.backbone(img)

        if self.training:
            # size augmentation during training, that is randomly drop grids
            base_length = base_features.size(1)
            features = []
            features_lengths = []
            rand_list_1 = np.random.rand(base_features.size(0), base_features.size(1))
            rand_list_2 = np.random.rand(base_features.size(0))
            for i in range(base_features.size(0)):
                if rand_list_2[i] > 0.20:
                    feature_i = base_features[i][np.where(rand_list_1[i] > 0.20 * rand_list_2[i])]
                    len_i = len(feature_i)
                    pads_i = torch.zeros(base_length - len_i,
                                         base_features.size(-1)).to(base_features.device)
                    feature_i = torch.cat([feature_i, pads_i], dim=0)
                else:
                    feature_i = base_features[i]
                    len_i = base_length
                features_lengths.append(len_i)
                features.append(feature_i)
            base_features = torch.stack(features, dim=0)
            base_features = base_features[:, :max(features_lengths), :]
            features_lengths = torch.tensor(features_lengths)
        else:
            features_lengths = torch.zeros(base_features.size(0), dtype=torch.long)
            features_lengths[:] = base_features.size(1)
        features = self.img_proj(base_features, features_lengths)
        return features

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, fixed_blocks):
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone.unfreeze_base(fixed_blocks)


class EncoderImage_VIT(nn.Module):
    def __init__(self, vit_type, emb_size, precomp_enc_type='basic', no_imgnorm=False):
        super(EncoderImage_VIT, self).__init__()
        self.vit_model = ViTModel.from_pretrained(vit_type)
        self.img_dim = self.vit_model.config.hidden_size
        self.img_proj = EncoderImage_ProjFeature(self.img_dim, emb_size,
                                                 precomp_enc_type, no_imgnorm)

    def forward(self, img, img_lengths=None):
        img = self.vit_model(img)
        img = img[0]
        if self.training:
            img_lengths = img.shape[1] * torch.ones(img.shape[0], dtype=torch.long)
        else:
            img, img_lengths = aug_vision(img)

        features = self.img_proj(img, img_lengths)
        return features


# Augmentation for vision
def aug_vision(imgs):
    lengths = []
    features = torch.Tensor(imgs.shape).to(imgs.device)
    for i in range(len(imgs)):
        img = imgs[i]
        num_regions = img.shape[0]
        rand_list = np.random.rand(num_regions)
        img = img[np.where(rand_list > 0.10)]
        lengths.append(img.shape[0])
        features[i, :img.shape[0], :] = img
    max_length = max(lengths)
    features = features[:, :max_length, :]

    return features, torch.LongTensor(lengths)