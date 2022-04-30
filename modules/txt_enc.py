import torch
import torch.nn as nn
from transformers import BertModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import l2norm
from modules.adcap import AP


# Bi-GRU based Language Enocder
class EncoderText_BiGRU(nn.Module):
    def __init__(self, vocab_size, word_dim, emb_size, num_layers,
                 use_bigru=False, no_txtnorm=False):
        super(EncoderText_BiGRU, self).__init__()
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.emb_size = emb_size
        self.no_txtnorm = no_txtnorm
        self.embed = nn.Embedding(self.vocab_size, self.word_dim)
        self.use_bigru = use_bigru
        self.rnn = nn.GRU(self.word_dim, self.emb_size, num_layers,
                          batch_first=True,
                          bidirectional=self.use_bigru)
        self.pool = AP(self.emb_size)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, txt, lengths):
        '''
           Extract sentence features
           input = [batch_size, seq_len]
           output = [batch_size, seq_len, emb_size]
        '''
        # embedding layer
        txt_emb = self.embed(txt)
        self.rnn.flatten_parameters()
        try:
            txt_emb = pack_padded_sequence(txt_emb, lengths, batch_first=True)
        except:
            txt_emb = pack_padded_sequence(txt_emb, lengths, enforce_sorted=False,
                                           batch_first=True)
        # bi rnn layers
        txt_emb, _ = self.rnn(txt_emb)
        txt_emb, length = pad_packed_sequence(txt_emb, batch_first=True)

        if self.use_bigru:
            txt_emb = (txt_emb[:, :, :txt_emb.size(2) // 2] + txt_emb[:, :, txt_emb.size(2) // 2:]) / 2

        # pooling operation
        txt_lengths = torch.LongTensor(length).to(txt_emb.device)
        txt_emb, _ = self.pool(txt_emb, txt_lengths)

        # normalization
        if not self.no_txtnorm:
            txt_emb = l2norm(txt_emb, dim=-1)

        return txt_emb, length


# BERT based Language Encoder
class EncoderText_BERT(nn.Module):
    def __init__(self, bert_type, emb_size, args, no_txtnorm=False):
        super(EncoderText_BERT, self).__init__()
        self.args = args
        self.bert_type = bert_type
        self.emb_size = emb_size
        self.no_txtnorm = no_txtnorm

        self.bert = BertModel.from_pretrained(self.bert_type)
        self.bert_dim = self.bert.config.hidden_size

        self.fc = nn.Linear(self.bert_dim, self.emb_size)
        self.pool = AP(self.emb_size)

    def forward(self, txt, lengths):
        batch_size, max_length = txt.shape
        txt_mask = torch.ones((txt.shape),
                              dtype=torch.int)
        txt_mask = (txt != 0).float()

        txt_emb = self.bert(txt, txt_mask).last_hidden_state
        txt_emb = self.fc(txt_emb)

        # pooling operation
        txt_lengths = torch.LongTensor(lengths).to(txt_emb.device)
        txt_emb, _ = self.pool(txt_emb, txt_lengths)

        # normalization
        if not self.no_txtnorm:
            txt_emb = l2norm(txt_emb, dim=-1)

        return txt_emb, lengths