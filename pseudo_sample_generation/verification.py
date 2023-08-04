'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from BLIP.models.blip_itm import blip_itm
import BLIP.utils as utils
from BLIP.utils import cosine_lr_schedule
from BLIP.data import create_dataset, create_sampler, create_loader
from BLIP.data.utils import save_result
import numpy


@torch.no_grad()
def evaluation(model, data_loader, device, config) :
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print_freq = 50
    
    result = []
        
    for image, text in metric_logger.log_every(data_loader, print_freq, header):        
        image = image.to(device)             
        output = model(image, text, match_head='itm')
        itm_score = torch.nn.functional.softmax(output,dim=1)[:,1]
        result.append(itm_score.cpu().detach())
    result = torch.cat(result, dim=0)
    return result

def merge(data, weight):
    for i in range(len(data)):
        data[i].append(weight[i].numpy())
    return data

def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    #### Dataset #### 
    print("Creating grounding datasets")
    datasets = create_dataset('grounding', config)   
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([datasets], [False], num_tasks, global_rank)         
    else:
        samplers = [None]
    
    test_loader = create_loader([datasets],samplers,
                                              batch_size=[config['batch_size_test']],
                                              num_workers=[4],is_trains=[False], 
                                              collate_fns=[None])[0]
    #### Model #### 
    print("Creating model")
    model = blip_itm(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'])

    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    start_time = time.time()
    itm_result = evaluation(model_without_ddp, test_loader, device, config)
    data = torch.load(config['test_file'])
    data = merge(data, itm_result)
    file_name = '{0}_{1}.pth'.format(args.dataset, args.modal)
    save_path = os.path.join(args.output_dir, file_name)
    torch.save(data, save_path)
    #torch.save(grounding_result, 'output/result.pth')        
    #result_file = save_result(grounding_result, args.result_dir, 'verify_result')  
                      
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str)) 
    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='pseudo_sample_generation/BLIP/configs/verify.yaml') 
    parser.add_argument('--output_dir', default='output/unc')
    parser.add_argument('--evaluate', action='store_true')      
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--modal', default='uni_modal')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--dataset', default='unc', type=str)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)