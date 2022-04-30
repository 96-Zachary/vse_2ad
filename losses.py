import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class ContrastiveLoss(nn.Module):
    def __init__(self, args, device, lamda=0.01, acc_mode='acc'):
        super(ContrastiveLoss, self).__init__()
        self.args = args
        self.device = device
        self.lamda = lamda
        self.acc_mode = acc_mode
        self.scale = args.scale
        self.const_num = args.const_num
        self.loss_func = F.cross_entropy

    def align(self, d):
        return d.diag().mean()

    def uniform(self, d):
        d = d - torch.eye(d.shape[0]).to(d.device) * d
        return d.mul(-1).exp().mean()

    def ADCTO(self, d, query, keys, acc_mode, scale, num_negs=None):
        """
            d: [batch_size, batch_size] <--> query x keys
            querys: [batch_size, emb_size]
            keys: [batch_size, emb_size]
        """
        # positive_logit = [batch_size, 1]
        positive_logit = torch.diag(d).unsqueeze(-1)
        if acc_mode == 'acc':
            align = self.align(d).detach().data.cpu().numpy()
            uniform = self.uniform(d).detach().data.cpu().numpy()
            tmp = align + uniform
            num_negs = max(int(np.cos(tmp ** scale) * d.shape[0]) + 1, 1)
            num_negs = min(num_negs, d.shape[0] - 1)
        elif acc_mode == 'constant':
            num_negs = 1 if not num_negs else num_negs
        elif acc_mode == 'random':
            num_negs = np.random.randint(1, d.shape[0])

        # mask = [batch_size, batch_size]
        mask = (torch.eye(d.size(0)) > .5).to(d.device)
        d = d.masked_fill(mask, 0)

        # sorted_idx = [batch_size, num_negs, emb_size]
        _, sorted_idx = torch.sort(d, dim=-1, descending=True)
        sorted_idx = sorted_idx[:, :num_negs]
        sorted_idx = sorted_idx.unsqueeze(-1).repeat(1, 1, keys.shape[-1])

        # negative_keys = [batch_size, num_negs, emb_size]
        negative_keys = torch.gather(keys.repeat(d.shape[0], 1).view(d.shape[0], d.shape[0], -1),
                                     1, sorted_idx)

        # negative_logits = [batch_size, num_negs]
        negative_logits = torch.matmul(query.unsqueeze(1),
                                       negative_keys.transpose(-2, -1)).squeeze(1) + self.args.margin

        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=d.device)

        return self.loss_func(logits / self.lamda, labels), num_negs

    def forward(self, img, txt, txt_lens):
        # cos_sim = [batch_size, batch_size]
        cos_sim = txt.mm(img.t())
        t2i_loss, t2i_num_negs = self.ADCTO(cos_sim, txt, img, self.acc_mode,
                                            self.scale, self.const_num)
        i2t_loss, i2t_num_negs = self.ADCTO(cos_sim.t(), img, txt, self.acc_mode,
                                            self.scale, self.const_num)
        return (t2i_loss + i2t_loss) / 2, [t2i_num_negs, i2t_num_negs]


class TripletLoss(nn.Module):
    def __init__(self, args, margin=0, max_violation=False):
        super(TripletLoss, self).__init__()
        self.args = args
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, img, txt, txt_lens):
        scores = img.mm(txt.t())

        # scores = [batch_size, batch_size]
        # diagonal = [batch_size, 1]
        diagonal = scores.diag().view(img.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum(), [1, 1]