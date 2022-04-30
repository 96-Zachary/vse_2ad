import os
import cv2
import json
import random
import numpy as np
from imageio import imread

import torch
import torch.utils.data as data

from PIL import Image
from transformers import ViTFeatureExtractor


# Dataset for region-level image and token-level text
class RegionLanguageDataset(data.Dataset):
    def __init__(self, data_path, data_split, vocab, tokenizer, args=None):
        self.vocab = vocab
        self.data_split = data_split
        self.data_path = data_path + '/precomp'
        self.args = args
        self.tokenizer = tokenizer

        # load captions
        self.captions = []
        with open(f'{self.data_path}/{data_split}_caps.txt', 'rb') as f:
            for line in f:
                self.captions.append(line.decode().strip())
        # load image regions features
        self.images = np.load(f'{self.data_path}/{data_split}_ims.npy')

        if len(self.images) != len(self.captions):
            self.im_div = 5
        else:
            self.im_div = 1
        if self.data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # image
        img_idx = index // self.im_div
        image = aug_vision(self.images[img_idx], self.data_split)

        # caption
        caption = self.captions[index]
        if self.args.txt_enc_type == 'rnn':
            tokens = self.tokenizer(str(caption).lower())
            caption = aug_language(tokens, self.vocab, self.data_split,
                                   self.args.txt_enc_type)
        elif self.args.txt_enc_type == 'bert':
            tokens = self.tokenizer.basic_tokenizer.tokenize(str(caption).lower())
            caption = aug_language(tokens, self.tokenizer, self.data_split,
                                   self.args.txt_enc_type)

        # image = [region_num, img_dim]
        # caption = [seq_len, seq_dim]
        return image, caption, index, img_idx

    def __len__(self):
        return len(self.captions)


# Dataset for grid-level image and token-level text
class GridLanguageDataset(data.Dataset):
    def __init__(self, data_path, data_split, vocab, tokenizer,
                 args=None, target_size=None, fextractor=None):
        self.vocab = vocab
        self.data_split = data_split
        self.data_path = data_path
        self.args = args
        self.tokenizer = tokenizer
        self.fextractor = fextractor

        # load captions
        self.captions = []
        with open(f'{self.data_path}/precomp/{data_split}_caps.txt', 'rb') as f:
            for line in f:
                self.captions.append(line.decode().strip())

        # load grid image
        # load id2path file
        with open(self.data_path + '/id_mapping.json', 'r') as f_mapping:
            self.id_to_path = json.load(f_mapping)
        with open(f'{self.data_path}/precomp/{self.data_split}_ids.txt', 'r') as f:
            image_ids = f.readlines()
            self.images = [int(x.strip()) for x in image_ids]
        assert 'backbone' in self.args.precomp_enc_type

        self.base_target_size = target_size
        self.crop_rate = 0.875
        self.train_scale_rate = 1
        self.base_target_size = int(self.base_target_size * self.args.input_scale_factor)
        self.pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])

        self.length = len(self.captions)
        num_images = len(self.images)
        if self.length != num_images:
            self.im_div = 5
        else:
            self.im_div = 1
        if self.data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # caption
        caption = self.captions[index]
        if self.args.txt_enc_type == 'rnn':
            tokens = self.tokenizer(str(caption).lower())
            caption = aug_language(tokens, self.vocab, self.data_split,
                                   self.args.txt_enc_type)
        elif self.args.txt_enc_type == 'bert':
            tokens = self.tokenizer.basic_tokenizer.tokenize(str(caption).lower())
            caption = aug_language(tokens, self.tokenizer, self.data_split,
                                   self.args.txt_enc_type)
        # image
        img_index = index // self.im_div
        if self.args.img_enc_type == 'butd':
            image_id = self.images[img_index]
            image_path = f'{self.data_path}/images/{self.id_to_path[str(image_id)]}'
            im_in = np.array(imread(image_path))
            processed_image = self._process_image(im_in)
            image = torch.Tensor(processed_image)
            image = image.permute(2, 0, 1)
        else:
            image_id = self.images[img_index]
            image_path = f'{self.data_path}/images/{self.id_to_path[str(image_id)]}'
            img = Image.open(image_path).convert('RGB') 
            image = self.fextractor(img, return_tensors='pt')
            image = image['pixel_values'].squeeze(0)

        return image, caption, index, img_index

    def __len__(self):
        return self.length

    def _process_image(self, im_in):
        """
            scaling, padding and data augmentation
        """
        if len(im_in.shape) == 2:
            im_in = im_in[:, :, :np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in),
                                   axis=2)
        im_in = im_in[:, :, ::-1]
        im = im_in.astype(np.float32, copy=True)
        target_size = self.base_target_size

        # random crop in train mode
        if self.data_split == 'train':
            crop_ratio = np.random.random() * 0.4 + 0.6
            crop_size_h = int(im.shape[0] * crop_ratio)
            crop_size_w = int(im.shape[1] * crop_ratio)
            processed_im = self._crop(im, crop_size_h, crop_size_w, random=True)
        else:
            processed_im = im

        # resize to the target resolution
        im_shape = processed_im.shape
        im_scale_x = float(target_size) / im_shape[1]
        im_scale_y = float(target_size) / im_shape[0]
        processed_im = cv2.resize(processed_im, None, None,
                                  fx=im_scale_x, fy=im_scale_y,
                                  interpolation=cv2.INTER_LINEAR)

        if self.data_split == 'train':
            if np.random.random() > 0.5:
                processed_im = self._hori_flip(processed_im)

        # normalization
        processed_im = self._image_norm(processed_im)

        return processed_im

    def _image_norm(self, im_in):
        im_in = im_in.astype(np.float32)
        im_in -= self.pixel_means
        return im_in

    @staticmethod
    def _crop(im, crop_size_h, crop_size_w, random):
        h, w = im.shape[0], im.shape[1]
        if random:
            if w - crop_size_w == 0:
                x_start = 0
            else:
                x_start = np.random.randint(w - crop_size_w, size=1)[0]
            if h - crop_size_h == 0:
                y_start = 0
            else:
                y_start = np.random.randint(h - crop_size_h, size=1)[0]
        else:
            x_start = (w - crop_size_w) // 2
            y_start = (h - crop_size_h) // 2
        cropped_im = im[y_start:y_start + crop_size_h, x_start:x_start + crop_size_w, :]

        return cropped_im

    @staticmethod
    def _hori_flip(im):
        im = np.fliplr(im).copy()
        return im


# Augmentation for vision and language
def aug_vision(image, data_split):
    if data_split == 'train':
        num_regions = image.shape[0]
        rand_list = np.random.rand(num_regions)
        image = image[np.where(rand_list > 0.20)]
    return torch.Tensor(image)


def aug_language(tokens, vocab, data_split, enc_type):
    if enc_type == 'rnn':
        out_tokens = ['<start>']
        if data_split == 'train':
            for i, token in enumerate(tokens):
                prob = random.random()
                if prob < 0.20:
                    # 50% randomly change token to mask token
                    prob /= 0.20
                    if prob < 0.50:
                        out_tokens.append('<mask>')
                        # 10% randomly change to random token
                    elif prob < 0.60:
                        out_tokens.append(random.choice(list(vocab.word2idx.keys())))
                    # 40% randomly remove the token
                else:
                    out_tokens.append(token)
        else:
            out_tokens += tokens
        out_tokens.append('<end>')
        tokens = [vocab(tok) for tok in out_tokens]
    else:
        out_tokens = []
        for i, token in enumerate(tokens):
            sub_tokens = vocab.wordpiece_tokenizer.tokenize(token)
            prob = random.random()
            if prob < 0.20 and data_split == 'train':
                prob /= 0.20
                if prob < 0.50:
                    for _ in sub_tokens:
                        out_tokens.append("[MASK]")
                elif prob < 0.60:
                    for sub_token in sub_tokens:
                        out_tokens.append(random.choice(list(vocab.vocab.keys())))
            else:
                for sub_token in sub_tokens:
                    out_tokens.append(sub_token)
        tokens = ['[CLS]'] + out_tokens + ['[SEP]']
        tokens = vocab.convert_tokens_to_ids(tokens)
    txt = torch.Tensor(tokens)
    return txt


def collate_sorted_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)

    # Merge sentencs
    txt_lengths = torch.LongTensor([len(cap) for cap in captions])
    txts = torch.zeros(len(captions), max(txt_lengths)).long()
    for i, cap in enumerate(captions):
        end = txt_lengths[i]
        txts[i, :end] = cap[:end]

    # Merge images
    # images : tuple --> [batch_size, region_num, img_dim]
    if len(images[0].shape) == 2:
        img_lengths = [len(img) for img in images]
        imgs = torch.zeros(len(img_lengths), max(img_lengths), images[0].size(-1))
        for i, img in enumerate(images):
            end = img_lengths[i]
            imgs[i, :end] = img[:end]
        img_lengths = torch.LongTensor(img_lengths)

        return imgs, img_lengths, txts, txt_lengths, ids
    else:
        imgs = torch.stack(images, 0)
        return imgs, txts, txt_lengths, ids


def collate_unsorted_fn(data):
    images, captions, ids, img_ids = zip(*data)

    # Merge sentencs
    txt_lengths = torch.LongTensor([len(cap) for cap in captions])
    txts = torch.zeros(len(captions), max(txt_lengths)).long()
    for i, cap in enumerate(captions):
        end = txt_lengths[i]
        txts[i, :end] = cap[:end]

    # Merge images
    # images : tuple --> [batch_size, region_num, img_dim]
    if len(images[0].shape) == 2:
        img_lengths = [len(img) for img in images]
        imgs = torch.zeros(len(img_lengths), max(img_lengths), images[0].size(-1))
        for i, img in enumerate(images):
            end = img_lengths[i]
            imgs[i, :end] = img[:end]
        img_lengths = torch.LongTensor(img_lengths)

        return imgs, img_lengths, txts, txt_lengths, ids
    else:
        imgs = torch.stack(images, 0)
        return imgs, txts, txt_lengths, ids


def get_dataloader(data_path, data_split, vocab, tokenizer, args, batch_size,
                   shuffle=True, number_workers=2):
    if args.precomp_enc_type == 'basic':
        dataset = RegionLanguageDataset(data_path, data_split, vocab, tokenizer, args)
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  pin_memory=True,
                                                  num_workers=number_workers,
                                                  collate_fn=collate_sorted_fn if args.txt_enc_type == 'rnn' else collate_unsorted_fn,
                                                  drop_last=True if data_split == 'train' else False
                                                  )
    else:
        if args.img_enc_type == 'vit':
            feature_extractor = ViTFeatureExtractor.from_pretrained(args.vit_type)
            target_size = 224
            dataset = GridLanguageDataset(data_path, data_split, vocab, tokenizer,
                                          args, target_size, feature_extractor)
        else:
            target_size = 256
            dataset = GridLanguageDataset(data_path, data_split, vocab, tokenizer, args, target_size)
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  pin_memory=True,
                                                  num_workers=number_workers,
                                                  collate_fn=collate_sorted_fn if args.txt_enc_type == 'rnn' else collate_unsorted_fn)

    return data_loader


def get_loaders(data_name, vocab, tokenizer, batch_size, workers, args, return_test=False, test_data='test'):
    dpath = os.path.join(args.data_path, args.data_name)
    fold_size = 5 * int(batch_size // 5)
    train_loader, val_loader = None, None
    if not return_test:
        train_loader = get_dataloader(dpath, 'train', vocab, tokenizer, args,
                                      batch_size, True, workers)
        val_loader = get_dataloader(dpath, 'dev', vocab, tokenizer, args,
                                    fold_size, False, workers)
    test_loader = get_dataloader(dpath, test_data, vocab, tokenizer, args,
                                 fold_size, False, workers)
    return train_loader, val_loader, test_loader