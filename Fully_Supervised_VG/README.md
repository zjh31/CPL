# Fully Supervised Training and Validation

This repository is the official Pytorch implementation of fully supervised visual grounding.

## Usage

### Dependencies
- Python 3.9.10
- PyTorch 1.9.0 + cu111 + cp39
- [Pytorch-Bert 0.6.2](https://pypi.org/project/pytorch-pretrained-bert/)
- Check [requirements.txt](requirements.txt) for other dependencies. 

### Data Preparation

1. You should put your dataset annotations (XXX.pth file) in `./data/` folder. The format of data is:
```
[
    [
        'XXX.jpg', #image name
        [x1, y1, x2, y2], # bounding box label
        query, #natural language query
    ]
]
```
2. You should put your image data in `./image_data/` folder. 

### Pretrained Checkpoints
1.You can download the DETR checkpoints from [detr_checkpoints](https://disk.pku.edu.cn:443/link/2BB0B32AF9FB5FEF7CBE443D5642A6B7). The checkpoint should be downloaded and move to the checkpoints directory.

```
mkdir checkpoints
mv detr-r50.pth ./checkpoints/
```

### Training and Evaluation

1.  Training. 
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env train.py --num_workers 8 --epochs 20 --batch_size 32 --lr 0.0001 --lr_bert 0.00001 --lr_visu_cnn 0.00001 --lr_visu_tra 0.00001 --lr_scheduler cosine --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model checkpoints/detr-r50.pth --bert_enc_num 12 --detr_enc_num 6 --max_query_len 20 --data_root ./image_data/ --train_ann_file [training file (./data/XXX.pth)] --val_ann_file [validation file (./data/xxx.pth)] --output_dir ./outputs/
    ```
    *Notably, if you use a smaller batch size, you should also use a smaller learning rate. Original learning rate is set for batch size 256(8GPU x 32).* 

2.  Evaluation.
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 28888 --use_env eval.py --num_workers 4 --batch_size 128 --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset unc --max_query_len 20 --data_root ./data/image_data --eval_model ./outputs/best_checkpoint.pth --eval_file [evaluation file (./data/xxx.pth)] --output_dir ./outputs/;
    ```
