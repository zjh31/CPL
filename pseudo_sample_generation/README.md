# Generating Pseudo-samples
## Usage
First, we adopt the [VinVL](https://github.com/microsoft/scene_graph_benchmark) to get the object proposals and their attributes.
-You can download the detection results from [XXXX](XXXXX) and put them in the right place.
```
mkdir pseudo_sample_generation/detction_result
mv detects.tar.gz ./pseudo_sample_generation/detction_result/
tar -zxvf detects.tar.gz
```
## Generation of PseudoQuery-Region pairs
Take the RefCOCO dataset as an example. Please download the images of RefCOCO to ```./image_data/other/images/``` first. We need off-the-shelf NLP processing toolbox [Stanford CoreNLP](XXXX) and [word2vec model](XXXX) to fliter nouns and attributes. We also need caption model of [BLIP](./pseudo_sample_generation/BLIP/README.md) to generation caption by object centric pipeline and relation aware pipeline.
```
CUDA_VISIBLE_DEVICES=0 python pseudo_sample_generation/generate.py --image_root ./image_data/other/images/ --output_dir ./output/unc --nlp_model_file [stanford-corenlp model path] --google_model_file [word2vec model path] --object_file ./pseudo_sample_generation/detection_result/coco_data.pth --dataset unc --image_size 384 --blip_caption_file [blip_caption_model_path]
```

## Uni-Modal Real Query Propagation

After obtaining PseudoQuery-Region pairs, we utilize uni-modal similaity score to propagate the region to real query and form RealQuery-Region pairs.

```
CUDA_VISIBLE_DEVICES=0 python pseudo_sample_generation/propagation.py --output_dir ./output/unc --dataset unc --data_root ./data --uni_modal ./output/unc/pseudo_template.pth --cross_modal ./output/unc/pseudo_object.pth
```

## Verification of pseudo samples
After generating diverse pseudo samples(PseudoQuery-Region pairs and RealQuery-Region pairs), we utilize Image-Text Matching module of [BLIP](./pseudo_sample_generation/BLIP/README.md) to verify the quality of pseudo samples. 

```
#verification of realquery-region pairs
CUDA_VISIBLE_DEVICES=0 python pseudo_sample_generation/verification.py --config ./pseudo_sample_generation/BLIP/configs/verify_uni_modal.yaml --output_dir output/unc --modal uni_modal --dataset unc

#verification of pseudoquery-region pairs
CUDA_VISIBLE_DEVICES=0 python pseudo_sample_generation/verification.py --config ./pseudo_sample_generation/BLIP/configs/verify_cross_modal.yaml --output_dir output/unc --modal cross_modal --dataset unc
```
