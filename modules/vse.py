import torch
import torch.nn as nn

from modules.txt_enc import EncoderText_BERT, EncoderText_BiGRU
from modules.img_enc import EncoderImage_ProjFeature, EncoderImage_BUTD, EncoderImage_VIT
from modules.resnet import ResNetFeatureExtractor


class VSE(nn.Module):
    def __init__(self, args):
        super(VSE, self).__init__()
        self.args = args
        self.img_enc_type = args.img_enc_type
        self.txt_enc_type = args.txt_enc_type
        self.margin = args.margin

        # visiual encoder
        if self.img_enc_type == 'project':
            self.img_enc = EncoderImage_ProjFeature(args.img_dim, args.emb_size,
                                                    precomp_enc_type=args.precomp_enc_type,
                                                    no_imgnorm=args.no_imgnorm)
            self.proj_params = list(self.img_enc.parameters())
        elif self.img_enc_type == 'butd':

            backbone_cnn = ResNetFeatureExtractor(self.args.backbone_weights_path,
                                                  fixed_blocks=2)
            backbone_cnn.load_state_dict(
                torch.load(f'./data/weights/{args.data_name}_{args.txt_enc_type}_backbone.pth'))
            self.img_enc = EncoderImage_BUTD(backbone_cnn, self.args.img_dim,
                                             self.args.emb_size,
                                             precomp_enc_type='backbone')
            self.backbone_params = list(self.img_enc.backbone.base.parameters()) + \
                                   list(self.img_enc.backbone.top.parameters())
            self.proj_params = list(self.img_enc.img_proj.parameters())
        else:
            self.img_enc = EncoderImage_VIT(args.vit_type, args.emb_size,
                                            args.precomp_enc_type, args.no_imgnorm)
            self.vit_params = list(self.img_enc.vit_model.embeddings.parameters()) + \
                              list(self.img_enc.vit_model.encoder.parameters())
            self.proj_params = list(self.img_enc.vit_model.pooler.parameters()) + \
                               list(self.img_enc.img_proj.parameters())

        # textual encoder
        if self.txt_enc_type == 'rnn':
            self.txt_enc = EncoderText_BiGRU(args.vocab_size, args.word_dim,
                                             args.emb_size, args.num_layers,
                                             args.use_bigru, args.no_txtnorm)
            self.rnn_params = list(self.txt_enc.parameters())

        elif self.txt_enc_type == 'bert':
            self.txt_enc = EncoderText_BERT(args.bert_type, args.emb_size,
                                            args, args.no_txtnorm)
            self.txt_enc_bert_params = list(self.txt_enc.bert.parameters())
            bert_params_ptr = [p.data_ptr() for p in self.txt_enc_bert_params]
            self.txt_enc_nobert_params = list()
            for p in list(self.txt_enc.parameters()):
                if p.data_ptr() not in bert_params_ptr:
                    self.txt_enc_nobert_params.append(p)
        self.txt_enc_params = list(self.txt_enc.parameters())
        self.img_enc_params = list(self.img_enc.parameters())
        self.enc_params = self.img_enc_params + self.txt_enc_params

    def make_data_parallel(self):
        self.img_enc = nn.DataParallel(self.img_enc)
        self.txt_enc = nn.DataParallel(self.txt_enc)
        self.data_parallel = True

    def freeze_backbone(self):
        if self.img_enc_type == 'butd':
            self.img_enc.freeze_backbone()

    def unfreeze_backbone(self, fixed_blocks):
        if self.img_enc_type == 'butd':
            self.img_enc.unfreeze_backbone(fixed_blocks)

    def forward(self, img, img_lengths, txt, txt_lengths):
        # img = [batch_size, n_region, img_dim]
        # txt = [batch_size, seq_len]

        # img_emb = [batch_size, n_region, emb_size]
        # txt_emb  = [batch_size, seq_len, emb_size]
        img_emb = self.img_enc(img, img_lengths)
        txt_emb, txt_lens = self.txt_enc(txt, txt_lengths)

        return img_emb, txt_emb, txt_lens