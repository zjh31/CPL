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
Take the RefCOCO dataset as an example. Please download the images of RefCOCO to ```./image_data/other/images/``` first.
```
CUDA_VISIBLE_DEVICES=0 python pseudo_sample_generation/generate.py --image_root ./image_data/other/images/ --output_dir ./output/unc --nlp_model_file [stanford-corenlp model path] --google_model_file [word2vec model path] --object_file ./pseudo_sample_generation/detection_result/coco_data.pth --dataset unc --image_size 384 --blip_caption_file [blip_caption_model_path]
```
