# -*- coding: utf-8 -*-

"""
ReferIt, UNC, UNC+ and GRef referring image segmentation PyTorch dataset.

Define and group batches of images, segmentations and queries.
Based on:
https://github.com/chenxi116/TF-phrasecut-public/blob/master/build_batches.py
"""

import os
import re
# import cv2
import sys
import json
import torch
import numpy as np
import os.path as osp
import scipy.io as sio
import torch.utils.data as data

sys.path.append('.')

from PIL import Image
from pytorch_pretrained_bert.tokenization import BertTokenizer
from utils.word_utils import Corpus


def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line  # reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples


## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


class DatasetNotFoundError(Exception):
    pass


class TransVGDataset(data.Dataset):
    def __init__(self, data_root, ann_file,
                 transform=None, return_idx=False, testmode=False,
                 split='train', max_query_len=128, lstm=False,
                 bert_model='bert-base-uncased'):
        self.images = []
        self.data_root = data_root
        self.ann = ann_file
        self.query_len = max_query_len
        self.lstm = lstm
        self.transform = transform
        self.testmode = testmode
        self.split = split
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.return_idx = return_idx

        assert self.transform is not None

        if split in ['train']:
            self.augment = True
        else:
            self.augment = False
        
        self.im_dir = data_root

        if not self.exists_dataset():
            # self.process_dataset()
            print('The dataset {} is not found!'.format(self.ann))
            exit(0)
        self.images += torch.load(self.ann)[:5000]

    def exists_dataset(self):
        return osp.exists(self.ann)

    def pull_item(self, idx):
        img_file, bbox, phrase = self.images[idx]
        bbox = np.array(bbox, dtype=int)
        img_path = osp.join(self.im_dir, img_file)
        img = Image.open(img_path).convert("RGB")

        bbox = torch.tensor(bbox)
        bbox = bbox.float()
        return img, phrase, bbox 

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, phrase, bbox = self.pull_item(idx)
        phrase = phrase.lower()
        input_dict = {'img': img, 'box': bbox, 'text': phrase}
        input_dict = self.transform(input_dict)
        img = input_dict['img']
        bbox = input_dict['box']
        phrase = input_dict['text']
        img_mask = input_dict['mask']

        if self.lstm:
            phrase = self.tokenize_phrase(phrase)
            word_id = phrase
            word_mask = np.array(word_id > 0, dtype=int)
        else:
            ## encode phrase to bert input
            examples = read_examples(phrase, idx)
            features = convert_examples_to_features(
                examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
            word_id = features[0].input_ids
            word_mask = features[0].input_mask
        return img, np.array(img_mask), np.array(word_id, dtype=int), np.array(word_mask, dtype=int), np.array(bbox, dtype=np.float32)