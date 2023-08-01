import json
import os.path

import cv2
import torch
import pdb
import _pickle as cPickle
from transformers import BertModel
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer as Bert
from tqdm import tqdm
import gensim.downloader
import numpy as np
from gensim.models import KeyedVectors
import argparse
import os.path as osp

parser = argparse.ArgumentParser(description='Data preparation')
parser.add_argument('--output_dir', default='output/referit/', type=str)
parser.add_argument('--google_model_file', default='/network_space/storage43/zhangjiahua/GoogleNews-vectors-negative300.bin.gz', type=str)
parser.add_argument('--dataset', default='referit', type=str)
parser.add_argument('--data_root', default='data/', type=str)
parser.add_argument('--cross_modal', default='/home/zhangjiahua/Code/Pseudo-Q/output/referit/pseudo_object.pth', type=str)
parser.add_argument('--uni_modal', default='/home/zhangjiahua/Code/Pseudo-Q/output/referit/pseudo_template.pth', type=str)
args = parser.parse_args()







def cal_iou(box1, box2):
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
    inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
            (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    IoU = inter / union
    if IoU >= 0.5:
        return 1
    else:
        return 0

def merge_pseudo(args):
    images = torch.load(args.uni_modal)
    if not os.path.exists(args.cross_modal):
        print("uni-modal setting!")
        return images
    print("cross-modal setting")
    obj_centric = torch.load(args.cross_modal)
    for key in images:
        for i in range(len(obj_centric[key])):
            obj_centric[key][i][0] = obj_centric[key][i][0][0]
            images[key].append(obj_centric[key][i])
    for key in obj_centric:
        if key not in images:
            images[key] = []
            for i in range(len(obj_centric[key])):
                obj_centric[key][i][0] = obj_centric[key][i][0][0]
                images[key].append(obj_centric[key][i])
    return images


def pre_process(data_file):
    data = torch.load(data_file)
    img_box_list = []
    text_info = []
    for i in range(len(data)):
        img = data[i][0][:-4]
        gt = data[i][1]
        text = data[i][2]
        if [img, gt] not in img_box_list:
            img_box_list.append([img, gt])
            text_info.append([img, gt, [text]])
        else:
            index = img_box_list.index([img, gt])
            text_info[index][2].append(text)
    return text_info

def propagation(text_info, images, google_model):
    data_info = []
    recall = 0
    for i in tqdm(range(len(text_info))):
        img_name = text_info[i][0]
        text_list = text_info[i][2]
        box = text_info[i][1]
        if img_name not in images:
            continue
        pseudo_text = images[img_name]
        max_score = -2
        tmp = []
        for text in text_list:
            vector1 = np.zeros(300)
            texts = text.split()
            len1 = 0.00001
            for word in texts:
                if google_model.__contains__(word):
                    vector1 += google_model[word]
                    len1 = len1 + 1
            vector1 = vector1 / len1
            for pseudo in pseudo_text:
                text1 = pseudo[0].split()
                vector2 = np.zeros(300)
                len2 = 0
                for txt in text1:
                    if google_model.__contains__(txt):
                        vector2 += google_model[txt]
                        len2 += 1
                vector2 = vector2 / (len2 + 0.00001)
                score = vector1.dot(vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
                if score > max_score:
                    max_score = score
                    tmp = pseudo
        if len(tmp) == 0:
            continue
        recall += cal_iou(tmp[2], box) * len(text_list)
        for text in text_list:
            img_file = img_name + ".jpg"
            data_info.append([img_file, tmp[2], text])
        data_info.append([img_file, tmp[2], tmp[0]])
    print(recall/(len(data_info)-len(text_info)))
    return data_info

def generate_pseudo_pairs(args):
    file_name = '{0}_train.pth'.format(args.dataset)
    data_dir = osp.join(args.data_root, args.dataset)
    data_file = osp.join(data_dir, file_name)
    process_data = pre_process(data_file)
    images = merge_pseudo(args)
    google_model = KeyedVectors.load_word2vec_format(args.google_model_file, binary=True)
    pseudo_data = propagation(process_data, images, google_model)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    torch.save(pseudo_data, os.path.join(args.output_dir, 'real_query_pair.pth'))
generate_pseudo_pairs(args)

