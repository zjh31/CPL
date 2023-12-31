# Generating Pseudo-samples
## Usage
First, we adopt the [VinVL](https://github.com/microsoft/scene_graph_benchmark) to get the object proposals and their attributes. You can download the detection results from [detection results](https://disk.pku.edu.cn:443/link/E714889E66D48F2D81820BCF76BD6EB0) and put them in the right place.
```
mkdir pseudo_sample_generation/detction_result
mv detection_result.zip ./pseudo_sample_generation/detction_result/
unzip ./pseudo_sample_generation/detction_result/detection_result.zip
```
## Generation of PseudoQuery-Region pairs
Take the RefCOCO dataset as an example. Please download the images of RefCOCO to ```./image_data/other/images/``` first. We need off-the-shelf NLP processing toolbox Stanford CoreNLP and Word2Vec model to fliter nouns and attributes. You can download the two off-the-shelf NLP processing toolbox from [Stanford CoreNLP](https://disk.pku.edu.cn:443/link/20AA2B9F8960DD3957B91DE2AD87D9EC) and [Word2Vec](https://disk.pku.edu.cn:443/link/986DF918F422EA96BF175EEABD7136C5). We also need caption model of [BLIP](./BLIP/README.md) to generation caption by object centric pipeline and relation aware pipeline.
```
CUDA_VISIBLE_DEVICES=0 python pseudo_sample_generation/generate.py --image_root ./image_data/other/images/ --output_dir ./output/unc --nlp_model_file [stanford-corenlp model path] --google_model_file [Word2Vec model path] --object_file ./pseudo_sample_generation/detection_result/coco_data.pth --dataset unc --image_size 384 --blip_caption_file [blip_caption_model_path]
```

## Uni-Modal Real Query Propagation

After obtaining PseudoQuery-Region pairs, we utilize uni-modal similaity score to propagate the region to real query and form RealQuery-Region pairs.

```
python pseudo_sample_generation/propagation.py --output_dir ./output/unc --dataset unc --data_root ./data --uni_modal ./output/unc/pseudo_template.pth --cross_modal ./output/unc/pseudo_object.pth
```

## Verification of pseudo samples
After generating diverse pseudo samples(PseudoQuery-Region pairs and RealQuery-Region pairs), we utilize Image-Text Matching module of [BLIP](./BLIP/README.md) to verify the quality of pseudo samples. 

```
#verification of realquery-region pairs
CUDA_VISIBLE_DEVICES=0 python pseudo_sample_generation/verification.py --config ./pseudo_sample_generation/BLIP/configs/verify_uni_modal.yaml --output_dir output/unc --modal uni_modal --dataset unc

#verification of pseudoquery-region pairs
CUDA_VISIBLE_DEVICES=0 python pseudo_sample_generation/verification.py --config ./pseudo_sample_generation/BLIP/configs/verify_cross_modal.yaml --output_dir output/unc --modal cross_modal --dataset unc
```
