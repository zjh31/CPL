import os
import os.path as osp
import numpy as np
import cv2
import json
import argparse
import torch
from template import template_pipeline, object_centric_pipeline, relation_aware_pipeline, extra_nn_jj
from stanfordcorenlp import StanfordCoreNLP
from gensim.models import KeyedVectors
from BLIP.models.blip import blip_decoder
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
parser = argparse.ArgumentParser(description='generate pseudo query')
parser.add_argument('--img_root', default='/network_space/storage43/ln_data/referit/ReferIt', type=str)
parser.add_argument('--output_dir', default='output/referit/', type=str)
parser.add_argument('--nlp_model_file', default='/home/zhangjiahua/Code/stanford-corenlp-4.4.0', type=str)
parser.add_argument('--google_model_file', default='/network_space/storage43/zhangjiahua/GoogleNews-vectors-negative300.bin.gz', type=str)
parser.add_argument('--object_file', default='pseudo_sample_generation/object_detect/referit.pth', type=str)
parser.add_argument('--dataset', default='referit', type=str)
parser.add_argument('--data_root', default='data/', type=str)
parser.add_argument('--image_size', default=384, type=int)
parser.add_argument('--blip_caption_file', default='/home/zhangjiahua/Code/X-VLM/BLIP/model_base_capfilt_large.pth', type=str)
args = parser.parse_args()
def load_image(args, img_file, device):
    img_path = osp.join(args.img_root, img_file)
    raw_image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image

def load_crop_image(args, img_file, box, device):
    img_path = osp.join(args.img_root, img_file)
    raw_image = Image.open(img_path).crop(box).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image

def generate_sample(args):
    data_dir = osp.join(args.data_root, args.dataset)
    file_name = '{0}_train.pth'.format(args.dataset)
    data_file = osp.join(data_dir, file_name)
    nlp = StanfordCoreNLP(args.nlp_model_file)
    google_model = KeyedVectors.load_word2vec_format(args.google_model_file, binary=True)
    nn_vec, jj_vec = extra_nn_jj(data_file, nlp, google_model)
    pseudo_template = template_pipeline(data_file, nlp, google_model, args.object_file, nn_vec, jj_vec)
    data_obj = object_centric_pipeline(data_file, nlp, google_model, args.object_file, nn_vec, jj_vec)
    data_rel = relation_aware_pipeline(data_file, nlp, google_model, args.object_file, nn_vec, jj_vec)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = blip_decoder(pretrained=args.blip_caption_file, image_size=args.image_size, vit='base')
    model.eval()
    model = model.to(device)
    
    pseudo_relation = blip_relation_aware(args, data_rel, model, device)
    pseudo_object = blip_object_centric(args, data_obj, model, device)
    os.mkdir(args.output_dir)
    torch.save(pseudo_template, osp.join(args.output_dir, 'pseudo_template.pth'))
    torch.save(pseudo_object, osp.join(args.output_dir, 'pseudo_object.pth'))
    torch.save(pseudo_relation, osp.join(args.output_dir, 'pseudo_relation.pth'))



def blip_object_centric(args, data, model, device):
    print("Utilzing object centric pipeline to generate pseudo queries.")
    for key in tqdm(data):
        img_file = key + '.jpg'
        for i in range(len(data[key])):
            prompt, _, box = data[key][i]
            image = load_crop_image(args, img_file, box, device)
            captions = model.generate(image, sample=True, num_beams=1, max_length=20, min_length=5, prompts=prompt)
            data[key][i][0] = captions
    return data
def blip_relation_aware(args, data, model, device):
    print("Utilzing relation aware pipeline to generate pseudo queries.")
    generate_data = []
    for key in tqdm(data):
        img_file = key + '.jpg'
        image = load_image(args, img_file, device)
        for i in range(len(data[key])):
            prompt, _, box = data[key][i]
            captions = model.generate(image, sample=True, num_beams=3, max_length=20, min_length=5, prompts=prompt)
            for caption in captions:
                generate_data.append([img_file, box, caption])
    return generate_data

generate_sample(args)



