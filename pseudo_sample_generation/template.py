import os
import re
import json
import pdb
from tqdm import tqdm
import torch
import numpy as np
import cv2
order = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'nineth', 'tenth',
         'eleventh', 'twelfth', 'thirteenth','fourteenth', 'fifteenth', 'sixteenth', 'seventeenth', 'eighteenth', 'nineteenth', 'twentieth',
         'twenty-first', 'twenty-second', 'twenty-third', 'twenty-fourth', 'twenty-fifth', 'twenty-sixth', 'twenty-seventh', 'twenty-eighth', 'twenty-ninth', 'thirtieth',
         'thirty-first', 'thirty-second', 'thirty-third', 'thirty-fourth', 'thirty-fifth', 'thirty-sixth', 'thirty-seventh', 'thirty-eighth', 'thirty-ninth', 'fortieth',]
pos_dict = {}
from gensim.models import KeyedVectors
for i in range(40):
    pos_dict[i] = order[i]
for i in range(40, 100):
    pos_dict[i] = 'far'


def cal_overlap(box1, box2):
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
    inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
    S = (box2[2] - box2[0]) * (box2[3] - box2[1])
    IoU = inter / S
    if IoU >= 0.75:
        return True
    else:
        return False


def extra_nn_jj(data_file, nlp, google_model):
    data = torch.load(data_file)
    nn_vector = {}
    jj_vector = {}
    nn_dict = {}
    jj_dict = {}
    for i in tqdm(range(len(data))):
        image = data[i][0][:-4]
        if image not in nn_dict:
            nn_dict[image] = []
            jj_dict[image] = []
        text = data[i][2]
        pos_tag = nlp.pos_tag(text)
        for word, tag in pos_tag:
            if tag == 'JJ' and word not in jj_dict[image]:
                jj_dict[image].append(word)
            if tag == 'NN' and word not in nn_dict[image]:
                nn_dict[image].append(word)
    for key in tqdm(nn_dict):
        vector1 = []
        vector2 = []
        for i in range(len(nn_dict[key])):
            if google_model.__contains__(nn_dict[key][i]):
                vector1.append(google_model[nn_dict[key][i]])
        vector1 = np.array(vector1)
        nn_vector[key] = vector1
        for i in range(len(jj_dict[key])):
            if google_model.__contains__(jj_dict[key][i]):
                vector2.append(google_model[jj_dict[key][i]])
        vector2 = np.array(vector2)
        jj_vector[key] = vector2
    return nn_vector, jj_vector

def add_horizontal(data):
    for key in tqdm(data.keys()):
        kinds = {}
        tmp = []
        for j in range(len(data[key])):
            name = data[key][j][0]
            if name in kinds:
                kinds[name].append(data[key][j])
            else:
                kinds[name] = []
                kinds[name].append(data[key][j])
        for kind in kinds.keys():
            if len(kinds[kind]) == 2:
                x1 = (kinds[kind][0][3][0] + kinds[kind][0][3][2])/2
                x2 = (kinds[kind][1][3][0] + kinds[kind][1][3][2])/2
                if x1 - x2 > 0:
                    kinds[kind][0][4]['pos'].append('right')
                    kinds[kind][1][4]['pos'].append('left')
                elif x2 - x1 > 0:
                    kinds[kind][0][4]['pos'].append('left')
                    kinds[kind][1][4]['pos'].append('right')
            elif len(kinds[kind]) >= 3:
                a = []
                for i in range(len(kinds[kind])):
                    a.append((kinds[kind][i][3][0] + kinds[kind][i][3][2])/2)
                a = np.array(a)
                index = np.argsort(a)
                mid = index[int(len(a)/2)]
                kinds[kind][mid][4]['pos'].append('middle')
                for i in range(len(kinds[kind])):
                    pos = index[i]
                    if i == 0:
                        kinds[kind][pos][4]['pos'].append('left')
                    elif i == len(a) - 1:
                        kinds[kind][pos][4]['pos'].append('right')
                    elif pos != mid:
                        kinds[kind][pos][4]['pos'].append(pos_dict[i] + ' left')
                        kinds[kind][pos][4]['pos'].append(pos_dict[len(a) - i] + ' right')
            tmp = tmp + kinds[kind]
        data[key] = tmp
    return data

def add_vertical(data):
    for key in tqdm(data.keys()):
        kinds = {}
        tmp = []
        for j in range(len(data[key])):
            name = data[key][j][0]
            if name in kinds:
                kinds[name].append(data[key][j])
            else:
                kinds[name] = []
                kinds[name].append(data[key][j])
        for kind in kinds.keys():
            if len(kinds[kind]) == 2:
                x1 = (kinds[kind][0][3][1] + kinds[kind][0][3][3]) / 2
                x2 = (kinds[kind][1][3][1] + kinds[kind][1][3][3]) / 2
                if x1 - x2 > 0:
                    kinds[kind][0][4]['pos'].append('bottom')
                    kinds[kind][1][4]['pos'].append('top')
                elif x2 - x1 > 0:
                    kinds[kind][1][4]['pos'].append('bottom')
                    kinds[kind][0][4]['pos'].append('top')
            elif len(kinds[kind]) >= 3:
                a = []
                for i in range(len(kinds[kind])):
                    a.append((kinds[kind][i][3][1] + kinds[kind][i][3][3]) / 2)
                a = np.array(a)
                index = np.argsort(a)
                mid = index[int(len(a) / 2)]
                kinds[kind][mid][4]['pos'].append('center')
                for i in range(len(kinds[kind])):
                    pos = index[i]
                    if i == 0:
                        kinds[kind][pos][4]['pos'].append('top')
                    elif i == len(a) - 1:
                        kinds[kind][pos][4]['pos'].append('bottom')
                    elif pos != mid:
                        kinds[kind][pos][4]['pos'].append(pos_dict[i] + ' top')
                        kinds[kind][pos][4]['pos'].append(pos_dict[len(a) - i] + ' bottom')
            tmp = tmp + kinds[kind]
        data[key] = tmp
    return data

def add_depth(data):
    for key in tqdm(data.keys()):
        kinds = {}
        tmp = []
        for j in range(len(data[key])):
            name = data[key][j][0]
            if name in kinds:
                kinds[name].append(data[key][j])
            else:
                kinds[name] = []
                kinds[name].append(data[key][j])
        for kind in kinds.keys():
            if len(kinds[kind]) == 2:
                s1 = (kinds[kind][0][3][2] - kinds[kind][0][3][0]) * (kinds[kind][0][3][3] - kinds[kind][0][3][1])
                s2 = (kinds[kind][1][3][2] - kinds[kind][1][3][0]) * (kinds[kind][1][3][3] - kinds[kind][1][3][1])
                if s1/s2 > 1.0:
                    kinds[kind][0][4]['pos'].append('front')
                    kinds[kind][0][4]['pos'].append('bigger')
                    kinds[kind][1][4]['pos'].append('behind')
                    kinds[kind][1][4]['pos'].append('smaller')
                elif s2/s1 > 1.0:
                    kinds[kind][0][4]['pos'].append('behind')
                    kinds[kind][1][4]['pos'].append('front')
                    kinds[kind][0][4]['pos'].append('smaller')
                    kinds[kind][1][4]['pos'].append('bigger')
            elif len(kinds[kind]) >= 3:
                a = []
                for i in range(len(kinds[kind])):
                    a.append((kinds[kind][i][3][2] - kinds[kind][i][3][0]) * (kinds[kind][i][3][3] - kinds[kind][i][3][1]))
                max_index = a.index(max(a))
                min_index = a.index(min(a))
                max_s = (kinds[kind][max_index][3][2] - kinds[kind][max_index][3][0]) * (kinds[kind][max_index][3][3] - kinds[kind][max_index][3][1])
                min_s = (kinds[kind][min_index][3][2] - kinds[kind][min_index][3][0]) * (kinds[kind][min_index][3][3] - kinds[kind][min_index][3][1])
                if max_s/min_s > 1.5:
                    kinds[kind][min_index][4]['pos'].append('behind')
                    kinds[kind][max_index][4]['pos'].append('front')
                    kinds[kind][min_index][4]['pos'].append('smaller')
                    kinds[kind][max_index][4]['pos'].append('bigger')
                    for k in range(len(a)):
                        sq = (kinds[kind][k][3][2] - kinds[kind][k][3][0]) * (kinds[kind][k][3][3] - kinds[kind][k][3][1])
                        if max_s/sq < 1.03 and k != max_index:
                            kinds[kind][k][4]['pos'].append('front')
                            kinds[kind][k][4]['pos'].append('bigger')
                        elif sq/min_s < 1.03 and k!= min_index:
                            kinds[kind][k][4]['pos'].append('behind')
                            kinds[kind][k][4]['pos'].append('smaller')
            tmp = tmp + kinds[kind]
        data[key] = tmp
    return data

def add_obj(data):
    for key in tqdm(data.keys()):
        tmp = []
        for j in range(len(data[key])):
            s = (data[key][j][3][2] - data[key][j][3][0]) * (data[key][j][3][3] - data[key][j][3][1])
            tmp.append(s)
        s_tmp = np.array(tmp)
        s_sort = np.argsort(s_tmp)[::-1]
        for i in range(len(s_sort)-1):
            for j in range(i+1, len(s_sort)):
                index1 = s_sort[i]
                index2 = s_sort[j]
                box1 = data[key][index1][3]
                box2 = data[key][index2][3]
                if cal_overlap(box1, box2):
                    sent = dict(object=data[key][index2][0], attri=data[key][index2][1]['attri'])
                    data[key][index1][5]['with'].append(sent)
    return data
    


def filt_obj(data, nn_dict, jj_dict, google_model, data_file, attribute=True, object_num=-1):
    unc = torch.load(data_file)
    images = {}
    for i in range(len(unc)):
        img = unc[i][0][:-4]
        images[img] = 1
    final_data = {}
    for key in tqdm(images):
        tmp = []
        for i in range(len(data[key])):
            word_list, score, box = data[key][i]
            if google_model.__contains__(word_list[-1]) and nn_dict[key].shape[0]:
                vector = google_model[word_list[-1]]
                norm = np.linalg.norm(vector) * np.linalg.norm(nn_dict[key], axis=1)
                score = max(np.matmul(nn_dict[key], vector) / norm)
                if score < 0.5:
                    continue
            else:
                continue
            dict = {}
            dict['attri'] = []
            pos = {}
            have = {}
            pos['pos'] = []
            have['with'] = []
            for j in range(len(word_list)-1):
                if google_model.__contains__(word_list[j]) and jj_dict[key].shape[0]:
                    vector2 = google_model[word_list[j]]
                    norm2 = np.linalg.norm(vector2) * np.linalg.norm(jj_dict[key], axis=1)
                    if max(np.matmul(jj_dict[key], vector2) / norm2) > 0.5:
                        dict['attri'].append(word_list[j])
            if not attribute:
                dict['attri'] = []
            tmp.append([word_list[-1], dict, score, box, pos, have])
        final_data[key] = tmp
    for key in final_data:
        tmp = final_data[key]
        sort_text = sorted(tmp, key=(lambda x: x[2]), reverse=True)
        if object_num == -1:
            final_data[key] = sort_text
        else:
            final_data[key] = sort_text[:object_num]
    return final_data

def filt_obj_refer(data):
    nn_file = 'D:/Git Code/Pseudo-Q/refer_nn_vec.pth'
    nn_dict = torch.load(nn_file)
    jj_file = 'D:/Git Code/Pseudo-Q/refer_jj_vec.pth'
    jj_dict = torch.load(jj_file)
    file = 'D:/Git Code/ReDETR/data/referit/referit_train.pth'
    unc = torch.load(file)
    images = {}
    for i in range(len(unc)):
        img = unc[i][0][:-4]
        images[img] = 1
    final_data = {}
    for key in tqdm(images):
        tmp = []
        for i in range(len(data[key])):
            word_list, score, box, attri = data[key][i]
            if google_model.__contains__(word_list) and nn_dict[key].shape[0]:
                vector = google_model[word_list]
                norm = np.linalg.norm(vector) * np.linalg.norm(nn_dict[key], axis=1)
                score = max(np.matmul(nn_dict[key], vector) / norm)
                if score < 0.5:
                    continue
            else:
                continue
            dict = {}
            dict['attri'] = []
            pos = {}
            have = {}
            pos['pos'] = []
            have['with'] = []
            for j in range(len(attri)):
                if google_model.__contains__(attri[j]) and jj_dict[key].shape[0]:
                    vector2 = google_model[attri[j]]
                    norm2 = np.linalg.norm(vector2) * np.linalg.norm(jj_dict[key], axis=1)
                    if max(np.matmul(jj_dict[key], vector2) / norm2) > 0.5:
                        dict['attri'].append(attri[j])
            tmp.append([word_list, dict, score, box, pos, have])
        final_data[key] = tmp
    return final_data

    

def generate_text(data, pos_nes=False):
    for key in tqdm(data):
        tmp = []
        for obj in data[key]:
            having = []
            nn = []
            name, attri, score, box, pos, catch = obj
            for i in range(len(catch['with'])):
                word, jj = catch['with'][i]['object'], catch['with'][i]['attri']
                if len(jj) == 0:
                    having.append(word)
                for k in range(len(jj)):
                    having.append(jj[k] + " " + word)
            if len(attri['attri']) == 0:
                nn.append(name)
            for i in range(len(attri['attri'])):
                nn.append(attri['attri'][i] + " " + name)
            tmp.append([nn, score, box, pos, having])
        data[key] = tmp
    for key in tqdm(data):
        tmp = []
        for obj in data[key]:
            name, score, box, pos, catch = obj
            for i in range(len(name)):
                text = name[i]
                if len(pos['pos']) == 0 or pos_nes:
                    tmp.append([text, score, box])
                for j in range(len(pos['pos'])):
                    word = name[i].split()[-1]
                    text = name[i].replace(word, pos['pos'][j] + " " + word)
                    #text = name[i].replace(name[i], pos['pos'][j] + " " + name[i])
                    tmp.append([text, score, box])
                for j in range(len(catch)):
                    text = name[i] + " with " + catch[j]
                    tmp.append([text, score, box])
                for j in range(len(pos['pos'])):
                    for k in range(len(catch)):
                        word = name[i].split()[-1]
                        text = name[i].replace(word, pos['pos'][j] + " " + word)
                        #text = name[i].replace(name[i], pos['pos'][j] + " " + word)
                        text = text + " with " + catch[k]
                        tmp.append([text, score, box])
        data[key] = tmp
    return data

def cal_num(data):
    total = 0
    for key in data:
        total += len(data[key])
    print(total)

def template_pipeline(data_file, nlp, google_model, object_file, nn_vec, jj_vec, object_num=-1):
    detects = torch.load(object_file)
    data = filt_obj(detects, nn_vec, jj_vec, google_model, data_file, attribute=True, object_num=object_num)
    data = add_horizontal(data)
    data = add_vertical(data)
    data = add_depth(data)
    data = add_obj(data)
    final_data = generate_text(data, pos_nes=True)
    return final_data


def object_centric_pipeline(data_file, nlp, google_model, object_file, nn_vec, jj_vec):
    detects = torch.load(object_file)
    data = filt_obj(detects, nn_vec, jj_vec, google_model, data_file, attribute=False, object_num=10)
    final_data = generate_text(data, pos_nes=False)
    return final_data




def relation_aware_pipeline(data_file, nlp, google_model, object_file, nn_vec, jj_vec):
    detects = torch.load(object_file)
    data = filt_obj(detects, nn_vec, jj_vec, google_model, data_file, attribute=False, object_num=10)
    data = add_horizontal(data)
    final_data = generate_text(data, pos_nes=False)
    return final_data












