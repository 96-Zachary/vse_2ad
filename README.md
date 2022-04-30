# Official PyTorch implementation of the paper "Improving Visual-Semantic Embeddings with Adaptive Context-aware Pooling and Adaptive Clustering Objective"

## 1 Introduction
### 1.1 Paper Descriptions

### 1.2 File Descriptions
```train.py``` for training the VSE+2AD model using various visual semnatic backbones on COCO and Flickr30K.

```eval.py``` for evaluting the pre-trained models on COCO and Flickr30K.

```arguments.py``` for controling parameters for training.

```modules``` includes various files for building VSE+2AD model, which are ```vse.py```, ```mlp.py```, ```txt_enc.py```, ```img_enc.py```,```resnet.py``` and ```adcap.py```.
```voab.py``` for building or loading vocabularies.

```logger.py``` generates a logger for logging the both training and evaluating information.

```utils.py``` includes some basic function tools.

```losses.py``` for defining loss module, including adcto.

## 2 Preparation
### 2.1 Environment
The key dependencies on Ubuntu 20 for both training and inference are as following:
- python 3.8.1
- pytorch 1.8.0
- Transformers 4.1.0

### 2.2 Data
The original and precomputed data for COCO and Flickr30K, the pretrained weights and vocabulary should be considerd in ```data``` file, which orgnized as following:
```
data
├── f30k # Flickr30K dataset
│   ├── precomp # pre-computed BUTD region features for Flickr30K
│   │      ├── train_ids.txt
│   │      ├── train_caps.txt
│   │      ├── ......
│   ├── images # original images
│   │      ├── xxx.jpg
│   │      └── ...
│   ├── id_mapping.json
│   │
│   │
├── coco # MS-COCO dataset
│   ├── precomp # pre-computed BUTD region features for MS-COCO
│   │      ├── train_ids.txt
│   │      ├── train_caps.txt
│   │      ├── ......
│   ├── images # original images
│   │      ├── train2014
│   │      │   ├── xxx.jpg
│   │      │   ├──......
│   │      ├── val2014
│   ├── id_mapping.json
│   │
│   │
├── vocab # the vocabulary file
│   ├── f30k_precomp_vocab.json
│   ├── coco_precomp_vocab.json
│   │
│   │
├── weights
│   ├── original_updown_backbone.pth # the BUTD CNN weights
```
The download links for original COCO/F30K images, precomputed BUTD features, and corresponding vocabularies are from the offical repo of [SCAN](https://github.com/kuanghuei/SCAN#download-data).

```weights/original_updowmn_backbone.pth``` is the pre-trained ResNet-101 weights from [Bottom-up Attention Model](https://github.com/peteanderson80/bottom-up-attention).


## 3 Training
Run ```train.py``` to evaluate specified models on either COCO and Flickr30K.

For training ```bigru``` as textual backbone and ```project``` refers to the BUTD features with simple projection on Flickr30K or COCO, using the following command:
```
python train.py --log_name bigru_project_f30k \
                --data_name f30k/coco \
                --precomp_enc_type basic \
                --batch_size 128 \
                --txt_enc_type rnn \
                --use_bigru \
                --img_enc_type project \
                --loss_type acc
```

For fine-tune ```bigru``` and ```butd``` on Flickr30K, using the following command:
```
python train.py --log_name bigru_project_f30k \
                --data_name f30k/coco \
                --precomp_enc_type backbone \
                --batch_size 128 \
                --txt_enc_type rnn \
                --use_bigru \
                --img_enc_type butd \
                --backbone_lr_factor 0.05 \
                --loss_type acc
```

For fine-tune ```bigru``` and ```vit``` on Flickr30K, using the following command:
```
python train.py --log_name bigru_project_f30k \
                --data_name f30k/coco \
                --precomp_enc_type backbone \
                --batch_size 128 \
                --txt_enc_type rnn \
                --use_bigru \
                --img_enc_type vit \
                --vit_type google/vit-base-patch16-224 \
                --backbone_lr_factor 0.05 \
                --loss_type acc
```

For fine-tune ```BERT``` as textual backbone, the visual backbone is as same as above, 
```
python train.py --... \
                --txt_enc_type bert \
                --bert_type bert-base-uncased \
                --...
```

## 4 Evaluation
Run ```eval.py``` to evaluate specified models on either COCO and Flickr30K. The supported language encoder are ```TName={bigru/BERT}```, and vision encoder are ```IName={project/butd/vit}```.

For evaluting pre-trained models on Flickr-30K, use the command: 
```
python eval.py --data_name f30k \
               --txt_enc_type TName \
               --img_enc_type IName
```
For evaluting pre-trained models on MS-COCO, use the command: 
```
python eval.py --data_name coco \
               --txt_enc_type TName \
               --img_enc_type IName
```
