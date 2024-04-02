# CPL
<p align="center"> <img src='docs/model.png' align="center" height="250px"> </p>

This repository is the official Pytorch implementation for ICCV2023 paper **Confidence-aware Pseudo-label Learning for Weakly Supervised Visual Grounding**.

**Please leave a <font color='orange'>STAR ⭐</font> if you like this project!**

## Contents

1. [Usage](#usage)
2. [Results](#results)
3. [Contacts](#contacts)
4. [Acknowledgments](#acknowledgments)

## Usage

### Dependencies
- Python 3.9.10
- PyTorch 1.9.0 + cu111 + cp39
- [Pytorch-Bert 0.6.2](https://pypi.org/project/pytorch-pretrained-bert/)
- Check [requirements.txt](requirements.txt) for other dependencies. 

### Data Preparation
1.You can download the images from the original source and place them in `./data/image_data` folder:
- RefCOCO and ReferItGame
- [Flickr30K Entities](http://shannon.cs.illinois.edu/DenotationGraph/#:~:text=make%20face-,Downloads,-Please%20fill%20in)

Finally, the `./data/` and `./image_data/` folder will have the following structure:

```angular2html
|-- data
      |-- flickr
      |-- gref
      |-- gref_umd
      |-- referit
      |-- unc
      |-- unc+
|-- image_data
   |-- Flickr30k
      |-- flickr30k-images
   |-- other
      |-- images
   |-- referit
      |-- images
```
- ```./data/```: Take the Flickr30K dataset as an example, ./data/flickr/ shoud contain files about the dataset's train/validation/test annotations and our generated pseudo-samples for this dataset. You can download these file from [data](https://disk.pku.edu.cn/link/AAA0C1C7831CB54DA1840C1FFA2B1BA2A7) and put them on the corresponding folder.
- ```./image_data/Flickr30k/flickr30k-images/```: Image data for the Flickr30K dataset, please download from this [link](http://shannon.cs.illinois.edu/DenotationGraph/#:~:text=make%20face-,Downloads,-Please%20fill%20in). Fill the form and download the images.
- ```./image_data/other/images/```: Image data for RefCOCO/RefCOCO+/RefCOCOg. 
- ```./image_data/referit/images/```: Image data for ReferItGame.

2. The generated pseudo region-query pairs can be download from [data](https://disk.pku.edu.cn:443/link/29582215396BA69326A34F6DD2B2956A) or you can generate pseudo samples follow [instructions](./pseudo_sample_generation/README.md).

Note that to train the model with pseudo samples for different dataset you should put the uncompressed pseudo sample files under the right folder ```./data/xxx/```. For example, put the ```unc/train_cross_modal.pth``` under ```./data/unc/```.

For generating pseudo-samples, we adopt the pretrained detector and attribute classifier from the [VinVL]. The pytorch implementation of this paper is available at [VinVL](https://github.com/microsoft/scene_graph_benchmark).


### Pretrained Checkpoints
1.You can download the DETR checkpoints from [detr_checkpoints](https://disk.pku.edu.cn:443/link/4E6B5343270CC07E52A88AA8A7A31CE8). These checkpoints should be downloaded and move to the checkpoints directory.

```
mkdir checkpoints
mv detr_checkpoints.tar.gz ./checkpoints/
tar -zxvf detr_checkpoints.tar.gz
```

2.Checkpoints that trained on our pseudo-samples can be downloaded from [Google Drive](https://drive.google.com/file/d/19IhMNEgGIl4qGPq7v0SsD8VZSucfmlXj/view?usp=drive_link). You can evaluate the checkpoints following the instruction right below.

```
mv cpl_checkpoints.tar.gz ./checkpoints/
tar -zxvf cpl_checkpoints.tar.gz
```
### Training and Evaluation

1.  Training on RefCOCO. 
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 28888 --use_env train.py --num_workers 8 --epochs 20 --batch_size 32 --lr 0.0001 --lr_bert 0.00001 --lr_visu_cnn 0.00001 --lr_visu_tra 0.00001 --lr_scheduler cosine --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model checkpoints/detr-r50-unc.pth --bert_enc_num 12 --detr_enc_num 6 --dataset unc --max_query_len 20 --data_root ./data/image_data --split_root ./data/ --output_dir ./outputs/unc/
    ```
    *Notably, if you use a smaller batch size, you should also use a smaller learning rate. Original learning rate is set for batch size 128(4GPU x 32).* 
    Please refer to [scripts/train.sh](scripts/train.sh) for training commands on other datasets. 

2.  Evaluation on RefCOCO.
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 28888 --use_env eval.py --num_workers 4 --batch_size 128 --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset unc --max_query_len 20 --data_root ./data/image_data --split_root ./data/ --eval_model ./checkpoints/unc_best_checkpoint.pth --eval_set testA --output_dir ./outputs/unc/testA/;
    ```
    Please refer to [scripts/eval.sh](scripts/eval.sh) for evaluation commands on other splits or datasets.

## Results

<table border="2">
    <thead>
        <tr>
            <th colspan=3> &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp RefCOCO </th>
            <th colspan=3> &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp RefCOCO+</th>
            <th colspan=3> &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp RefCOCOg</th>
            <th colspan=1> ReferItGame</th>
            <th colspan=1> Flickr30K</th>
        </tr>
    </thead>
    <tbody>
    <tr>    
            <td>val</td>
            <td>testA</td>
            <td>testB</td>
            <td>val</td>
            <td>testA</td>
            <td>testB</td>
            <td>g-val</td>
            <td>u-val</td>
            <td>u-test</td>
            <td>test</td>
            <td>test</td>
        </tr>
    </tbody>
    <tbody>
    <tr>
            <td>70.67</td>
            <td>74.58</td>
            <td>67.19</td>
            <td>51.81</td>
            <td>58.34</td>
            <td>46.17</td>
            <td>57.04</td>
            <td>60.21</td>
            <td>60.12</td>
            <td>45.23</td>
            <td>63.87</td>
        </tr>
    </tbody>
</table>

## Contacts
zhangjiahua at stu dot pku dot edu dot cn

Any discussions or concerns are welcomed!

### Acknowledge
This codebase is partially based on [Pseudo-Q](https://github.com/LeapLabTHU/Pseudo-Q), [BLIP](https://github.com/salesforce/BLIP) and [VinVL](https://github.com/microsoft/scene_graph_benchmark).
