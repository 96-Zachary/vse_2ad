import time
import argparse
import numpy as np

import torch

from evaluation import i2t, t2i
from modules.vse import VSE
from vocab import load_vocab, get_vocab
from dataloader import get_loaders
from logger import get_logger

def rank_recall(d):
    i2t_r = i2t(d)
    t2i_r = t2i(d)
    rsum = sum(i2t_r) + sum(t2i_r)
    return i2t_r, t2i_r, rsum

def evaluate(model, iterator, logger, args, fold5=False, restore=False):
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
                
    if not fold5:
        img_embs = torch.cat([img_embs[i].unsqueeze(0) for i in range(0, len(img_embs), 5)])
        # [img_num, txt_num]
        d = img_embs.mm(cap_embs.t()).detach().data.numpy()
        i2t_r, t2i_r, rsum = rank_recall(d)

    else:
        res = []
        for i in range(5):
            img_embs_shard = img_embs[i*5000:(i+1)*5000:5]
            cap_embs_shard = cap_embs[i*5000:(i+1)*5000]
            d = img_embs_shard.mm(cap_embs_shard.t()).detach().data.numpy()
            i2t_r, t2i_r, rsum = rank_recall(d)
            res.append([i2t_r[0], i2t_r[1], i2t_r[2], t2i_r[0], t2i_r[1], t2i_r[2], rsum])
        mean_res = np.array(res).mean(axis=0).flatten()
        i2t_r, t2i_r, rsum = mean_res[0:3], mean_res[3:6], mean_res[-1]
    
    logger.info('Val_Rsum: {0:.2f} | Type: single'.format(rsum))
    logger.info('Text-Image Retrieval | R1:{0:.2f}, R5:{1:.2f}, R10:{2:.2f}'.format(*t2i_r))
    logger.info('Image-Text Retrieval | R1:{0:.2f}, R5:{1:.2f}, R10:{2:.2f}'.format(*i2t_r))       
    
    return i2t_r, t2i_r, rsum, d

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='coco', type=str)
    parser.add_argument('--txt_enc_type', default='bigru', type=str)
    parser.add_argument('--img_enc_type', default='vit', type=str)
    parser.add_argument('--log_path', default='./runs/', type=str)
    args = parser.parse_args([])
    
    args.log_name = f'test_{args.txt_enc_type}_{args.img_enc_type}'
    logger = get_logger(args, args.log_name)
    logger.info(f'Initialize logger file.')
    
    checkpoint_path = [f'./runs/{args.data_name}/params/{args.txt_enc_type}_{args.img_enc_type}_{args.data_name}/checkpoint.pth']
    
    # load checkpoint
    store_matrixs = []
    for path in checkpoint_path:
        checkpoint = torch.load(path)
        args = checkpoint['args']
        
        # load vocabulary
        vocab, tokenizer = get_vocab(args.txt_enc_type, args.vocab_path,
                                     args.data_name, args.bert_type)
        
        # initlize vse model
        model = VSE(args).to(args.device)
        model.load_state_dict(checkpoint['model'])
        
        logger.info(f'Model Structure: ')
        logger.info(f'Model Structure: {model}')
        
        # load test dataloader 
        if args.data_name == 'coco':
            logger.info('For MS-COCO 5-fold 1K:')
            _, _, test_loader = get_loaders(args.data_name, vocab, tokenizer,
                                            args.batch_size, args.num_workers, 
                                            args, return_test=True, test_data='testall')
            _, _, _, d = evaluate(model, test_loader, logger, args, fold5=True)
            store_matrixs.append(d)
            logger.info('For MS-COCO 5K:')
            _, _, _, d = evaluate(model, test_loader, logger, args)
            store_matrixs.append(d)
        else:
            logger.info('For Filckr30K:')
            _, _, test_loader = get_loaders(args.data_name, vocab, tokenizer,
                                            args.batch_size, args.num_workers, 
                                            args, return_test=True)
            _, _, _, d = evaluate(model, test_loader, logger, args)
            store_matrixs.append(d)
