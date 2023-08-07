#!/bin/sh

### RefCOCO/unc
python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env train.py --num_workers 8 --epochs 20 --batch_size 32 --lr 0.0001 --lr_bert 0.00001 --lr_visu_cnn 0.00001 --lr_visu_tra 0.00001 --lr_scheduler cosine --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model checkpoints/detr-r50-unc.pth --bert_enc_num 12 --detr_enc_num 6 --dataset unc --max_query_len 20 --data_root ./data/image_data --split_root ./data/ --output_dir ./outputs/unc/;

#### RefCOCO+/unc+
#python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env train.py --num_workers 8 --epochs 20 --batch_size 32 --lr 0.0001 --lr_bert 0.00001 --lr_visu_cnn 0.00001 --lr_visu_tra 0.00001 --lr_scheduler cosine --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model checkpoints/detr-r50-unc.pth --bert_enc_num 12 --detr_enc_num 6 --dataset unc+ --max_query_len 20 --data_root ./data/image_data --split_root ./data/ --output_dir ./outputs/unc+/;

#### RefCOCO gref
#python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env train.py --num_workers 8 --epochs 20 --batch_size 32 --lr 0.0001 --lr_bert 0.00001 --lr_visu_cnn 0.00001 --lr_visu_tra 0.00001 --lr_scheduler cosine --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model checkpoints/detr-r50-gref.pth --bert_enc_num 12 --detr_enc_num 6 --dataset gref --max_query_len 20 --data_root ./data/image_data --split_root ./data/ --output_dir ./outputs/gref/;

#### RefCOCO/gref_umd
#python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env train.py --num_workers 8 --epochs 20 --batch_size 32 --lr 0.0001 --lr_bert 0.00001 --lr_visu_cnn 0.00001 --lr_visu_tra 0.00001 --lr_scheduler cosine --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model checkpoints/detr-r50-gref.pth --bert_enc_num 12 --detr_enc_num 6 --dataset gref_umd --max_query_len 20 --data_root ./data/image_data --split_root ./data/ --output_dir ./outputs/gref_umd/;

#### ReferItGame/referit
#python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env train.py --num_workers 8 --epochs 20 --batch_size 32 --lr 0.0001 --lr_bert 0.00001 --lr_visu_cnn 0.00001 --lr_visu_tra 0.00001 --lr_scheduler cosine --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model checkpoints/detr-r50-referit.pth --bert_enc_num 12 --detr_enc_num 6 --dataset referit --max_query_len 20 --data_root ./data/image_data --split_root ./data/ --output_dir ./outputs/referit/;

#### Flickr
#python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env train.py --num_workers 8 --epochs 20 --batch_size 32 --lr 0.0001 --lr_bert 0.00001 --lr_visu_cnn 0.00001 --lr_visu_tra 0.00001 --lr_scheduler cosine --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model checkpoints/detr-r50-referit.pth --bert_enc_num 12 --detr_enc_num 6 --dataset flickr --max_query_len 20 --data_root ./data/image_data --split_root ./data/ --output_dir ./outputs/flickr/;
