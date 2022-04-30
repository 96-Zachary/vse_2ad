import os
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_

from dataloader import get_loaders
from logger import get_logger
from arguments import get_args
from vocab import load_vocab, get_vocab
from modules.vse import VSE
from evaluation import i2t, t2i
from losses import ContrastiveLoss, TripletLoss
from optimizer import adjust_learning_rate

def count_params(model):
    model_parameters = model.parameters()
    params = sum([np.prod(p.size()) for p in model_parameters if p.requires_grad])
    return params


def train(model, epoch, iterator, optimizer, loss_func, logger, length, args):
    epoch_start = time.time()
    epoch_loss = 0.0

    model.train()
    num_negs = []
    for step, data in enumerate(iterator):
        try:
            img, img_lengths, txt, txt_lengths, _ = data
        except:
            img, txt, txt_lengths, _ = data
            img_lengths = None
        if torch.cuda.is_available():
            img = img.to(args.device)
            txt = txt.to(args.device)

        # calculate loss and optimize
        optimizer.zero_grad()

        # model prediction
        # img_emb = [batch_size, n_region, emb_size]
        # txt_emb = [batch_size, seq_len, emb_size]
        img_emb, txt_emb, txt_lens = model(img, img_lengths, txt, txt_lengths)
        loss, tmp_num_negs = loss_func(img_emb, txt_emb, txt_lens)
        num_negs.extend(tmp_num_negs)

        loss.backward()
        if args.grad_clip > 0:
            clip_grad_norm_(model.enc_params, args.grad_clip)
        optimizer.step()

        epoch_loss += loss.item()

        if (step + 1) % args.log_step == 0:
            step_time = time.time() - epoch_start
            mins, secs = int(step_time // 60), int(step_time % 60)
            logger.info('Step: {0}/{1} | Loss: {2:.2f} | Time: {3}m {4}s | Step_Negs: {5:.2f}'.format(step + 1,
                                                                                                      length,
                                                                                                      epoch_loss / (
                                                                                                                  step + 1),
                                                                                                      mins, secs,
                                                                                                      np.mean(num_negs[
                                                                                                              step + 1 - args.log_step:step + 1])))

    epoch_time = time.time() - epoch_start
    mins, secs = int(epoch_time // 60), int(epoch_time % 60)
    logger.info('Train_Loss: {1:.3f} | Train_time: {2}m {3}s | Epoch_Negs: {4:.2f}'.format(epoch,
                                                                                           epoch_loss / length,
                                                                                           mins, secs,
                                                                                           np.mean(num_negs)))

    return epoch_loss / length


def evaluate(model, epoch, iterator, logger, args):
    epoch_start = time.time()
    model.eval()
    img_embs, cap_embs, cap_lens = None, None, None
    with torch.no_grad():
        for i, data in enumerate(iterator):
            try:
                img, img_lengths, txt, txt_lengths, ids = data
            except:
                img, txt, txt_lengths, ids = data
                img_lengths = None
            max_seq_length = max(txt_lengths)
            ids = np.array(ids)
            if torch.cuda.is_available():
                img = img.to(args.device)
                txt = txt.to(args.device)
            img_emb, txt_emb, txt_lens = model(img, img_lengths, txt, txt_lengths)
            if img_embs is None:
                if img_emb.dim() == 3:
                    img_embs = torch.zeros((len(iterator.dataset),
                                            img_emb.size(1),
                                            img_emb.size(2)))
                    cap_embs = torch.zeros((len(iterator.dataset),
                                            max_seq_length,
                                            txt_emb.size(2)))
                else:
                    img_embs = torch.zeros((len(iterator.dataset),
                                            img_emb.size(1)))
                    cap_embs = torch.zeros((len(iterator.dataset),
                                            txt_emb.size(1)))
                cap_lens = [0] * len(iterator.dataset)
            # cache embeddings
            img_embs[ids] = img_emb.data.cpu()
            if cap_embs.dim() == 3:
                cap_embs[ids, :max(txt_lengths), :] = txt_emb.data.cpu()
            else:
                cap_embs[ids, :] = txt_emb.data.cpu()
            for j, nid in enumerate(ids):
                cap_lens[nid] = txt_lens[j]

        img_embs = torch.cat([img_embs[i].unsqueeze(0) for i in range(0, len(img_embs), 5)])
        d = img_embs.mm(cap_embs.t()).detach().data.numpy()

    i2t_r = i2t(d)
    t2i_r = t2i(d)
    rsum = sum(i2t_r) + sum(t2i_r)

    epoch_time = time.time() - epoch_start
    mins, secs = int(epoch_time // 60), int(epoch_time % 60)

    logger.info('Val_Rsum: {0:.2f} | Val_Time: {1}m {2}s'.format(rsum, mins, secs))
    logger.info('Text-Image Retrieval | R1:{0:.2f}, R5:{1:.2f}, R10:{2:.2f}'.format(*t2i_r))
    logger.info('Image-Text Retrieval | R1:{0:.2f}, R5:{1:.2f}, R10:{2:.2f}'.format(*i2t_r))
    return i2t_r, t2i_r, rsum


def save_checkpoint(state, filename='checkpoint.pth', prefix=''):
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    torch.save(state, os.path.join(prefix, filename))

if __name__ == '__main__':
    # Load init arguments for training
    args = get_args()
    if torch.cuda.is_available():
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cudnn.benchmark = True

    # Initialize logger for record
    logger = get_logger(args, args.log_name)
    logger.info('Using txt-enc:"{0}" and img-enc:"{1}" for image-text retrieval in dataset:"{2}"'.format(args.txt_enc_type,
                                                                                                         args.img_enc_type,
                                                                                                         args.data_name))

    # Load precomputed vocabulary
    vocab, tokenizer = get_vocab(args.txt_enc_type, args.vocab_path,
                                 args.data_name, args.bert_type)

    # Logging the parameters in the record file
    args.vocab_size = len(vocab)
    logger.info('Parameters list: {0}'.format(args))

    # Load dataset loader
    train_loader, val_loader, test_loader = get_loaders(args.data_name, vocab, tokenizer,
                                                        args.batch_size, args.num_workers, args)
    logger.info('The iterator number of train/valid/test is {0}/{1}/{2}.'.format(len(train_loader),
                                                                                 len(val_loader),
                                                                                 len(test_loader)))

    decay_factor = 1e-4
    model = VSE(args).to(args.device)
    loss_func = ContrastiveLoss(args, args.device, args.tau)
    if args.txt_enc_type == 'rnn':
        if args.img_enc_type == 'project':
            optimizer = torch.optim.AdamW([{'params': model.rnn_params+model.proj_params,
                                            'lr': args.learning_rate}],
                                          lr=args.learning_rate)
        elif args.img_enc_type == 'butd':
            optimizer = torch.optim.AdamW([{'params': model.rnn_params,
                                            'lr': args.learning_rate},
                                           {'params': model.backbone_params,
                                            'lr': args.learning_rate * args.backbone_lr_factor},
                                           {'params': model.proj_params,
                                            'lr': args.learning_rate}],
                                          lr=args.learning_rate, weight_decay=decay_factor)
        else:
            optimizer = torch.optim.AdamW([{'params': model.rnn_params,
                                            'lr': args.learning_rate},
                                           {'params': model.vit_params,
                                            'lr': args.learning_rate * args.backbone_lr_factor},
                                           {'params': model.proj_params,
                                            'lr': args.learning_rate}],
                                          lr=args.learning_rate)
    if args.txt_enc_type == 'bert':
        if args.img_enc_type == 'project':
            optimizer = torch.optim.AdamW([{'params': model.proj_params,
                                            'lr': args.learning_rate},
                                           {'params': model.txt_enc_bert_params,
                                            'lr': args.learning_rate * 0.05},
                                           {'params': model.txt_enc_nobert_params,
                                            'lr': args.learning_rate}],
                                          lr=args.learning_rate, weight_decay=decay_factor)
        elif args.img_enc_type == 'butd':
            optimizer = torch.optim.AdamW([{'params': model.backbone_params,
                                            'lr': args.learning_rate * args.backbone_lr_factor},
                                           {'params': model.proj_params,
                                            'lr': args.learning_rate},
                                           {'params': model.txt_enc_bert_params,
                                            'lr': args.learning_rate * 0.1},
                                           {'params': model.txt_enc_nobert_params,
                                            'lr': args.learning_rate}],
                                          lr=args.learning_rate, weight_decay=decay_factor)
        else:
            optimizer = torch.optim.AdamW([{'params': model.txt_enc_bert_params,
                                            'lr': args.learning_rate * 0.1},
                                           {'params': model.txt_enc_nobert_params,
                                            'lr': args.learning_rate},
                                           {'params': model.vit_params,
                                            'lr': args.learning_rate * args.backbone_lr_factor},
                                           {'params': model.proj_params,
                                            'lr': args.learning_rate}],
                                          lr=args.learning_rate, weight_decay=decay_factor)

    best_rsum = 0.0
    for epoch in tqdm(range(args.epochs)):
        adjust_learning_rate(args, optimizer, epoch)
        logger.info('Epoch: {0}'.format(epoch))

        if args.precomp_enc_type == 'backbone':
            model.freeze_backbone()
            if epoch < args.embedding_warmup_epochs:
                model.freeze_backbone()
            else:
                model.unfreeze_backbone(0)

        train(model, epoch, train_loader, optimizer, loss_func, logger, len(train_loader), args)
        _, _, rsum = evaluate(model, epoch, test_loader, logger, args)

        # save checkpoint
        save_checkpoint({
            'model': model.state_dict(),
            'rsum': rsum,
            'args': args
        }, filename='checkpoint.pth',
            prefix=f'{args.log_path}{args.data_name}/params/{args.log_name}')

        best_rsum = max(rsum, best_rsum)
        logger.info('\n')