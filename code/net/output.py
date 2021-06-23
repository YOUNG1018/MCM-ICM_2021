# coding=utf-8
import os
import json
import csv
import argparse
import pandas as pd
import numpy as np
from math import ceil
from tqdm import tqdm
import pickle
import shutil

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torchvision import datasets, models
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from transforms import transforms
from models.LoadModel import MainModel
from utils.dataset_DCL import collate_fn4train, collate_fn4test, collate_fn4val, dataset
from config import LoadConfig, load_data_transformers
from utils.test_tool import set_text, save_multi_img, cls_base_acc

import pdb

import warnings

warnings.filterwarnings('ignore')

os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def parse_args():
    parser = argparse.ArgumentParser(description='dcl parameters')
    parser.add_argument('--data', dest='dataset',
                        default='CUB', type=str)
    parser.add_argument('--backbone', dest='backbone',
                        default='resnet50', type=str)
    parser.add_argument('--b', dest='batch_size',
                        default=16, type=int)
    parser.add_argument('--nw', dest='num_workers',
                        default=16, type=int)
    parser.add_argument('--ver', dest='version',
                        default='val', type=str)
    parser.add_argument('--save', dest='resume',
                        default=None, type=str)
    parser.add_argument('--size', dest='resize_resolution',
                        default=512, type=int)
    parser.add_argument('--crop', dest='crop_resolution',
                        default=448, type=int)
    parser.add_argument('--ss', dest='save_suffix',
                        default=None, type=str)
    parser.add_argument('--acc_report', dest='acc_report',
                        action='store_true')
    parser.add_argument('--swap_num', default=[7, 7],
                        nargs=2, metavar=('swap1', 'swap2'),
                        type=int, help='specify a range')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    args.version = 'test'
    if args.save_suffix == '':
        raise Exception('**** miss --ss save suffix is needed. ')

    Config = LoadConfig(args, args.version)
    transformers = load_data_transformers(args.resize_resolution, args.crop_resolution, args.swap_num)
    data_set = dataset(Config, anno=Config.val_anno if args.version == 'val' else Config.test_anno,
                       swap=transformers["None"], totensor=transformers['test_totensor'], test=True)

    dataloader = torch.utils.data.DataLoader(data_set,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             collate_fn=collate_fn4test)

    setattr(dataloader, 'total_item_len', len(data_set))

    cudnn.benchmark = True

    model = MainModel(Config)
    model_dict = model.state_dict()
    pretrained_dict = torch.load(args.resume)
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.cuda()
    model = nn.DataParallel(model)

    model.train(False)

    flag = 0

    with torch.no_grad():

        for batch_cnt_val, data_val in enumerate(dataloader):

            inputs, labels, img_name = data_val
            # print("label:", labels)
            inputs = Variable(inputs.cuda())
            labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())

            outputs = model(inputs)
            outputs_pred = outputs[0] + outputs[1][:, 0:Config.numcls] + outputs[1][:, Config.numcls:2 * Config.numcls]

            # print("= "*10)
            # print("outputs:", outputs)
            # print("= "*10)
            # print("outputs_pred:\n", outputs_pred.cpu().numpy())

            if flag == 0:
                scores = outputs_pred.cpu().numpy()
                flag = 1
            else:
                pred = outputs_pred.cpu().numpy()
                print(pred.shape, scores.shape)
                scores = np.vstack((scores, pred))

            # outputs_pred.cpu().numpy().tolist()

    # File operations

    # to CSV
    column = ['S_0', 'S_1', 'S_2', 'S_3', 'S_4']
    out = pd.DataFrame(columns=column, data=scores)
    out.to_csv('./pred_all.csv')

    # to PKL
    # with open('./pred_all.pkl', 'wb') as f:
    #     pickle.dump(scores, f)
    # print(scores)
