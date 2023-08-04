import json
import os
import math
import random
from random import random as rand
import torch
from torch.utils.data import Dataset

from torchvision.transforms.functional import hflip, resize

from PIL import Image
from BLIP.data.utils import pre_caption

class grounding(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=20):
        self.ann = []
        self.ann += torch.load(ann_file)
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words       
        
    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        img_file, box, text = ann
        image_path = os.path.join(self.image_root, img_file)
        image = Image.open(image_path).crop(box).convert('RGB')
        image = self.transform(image)
        text = pre_caption(text, self.max_words)
        return image, text
        


