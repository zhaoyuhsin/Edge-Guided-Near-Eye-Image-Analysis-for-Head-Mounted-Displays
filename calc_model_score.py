#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys, yaml, time
import copy
import torch
import pickle, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from models.RITnet_v2 import DenseNet2D
from models.RITnet_v1 import DenseNet2D as RITNet
from models.deepvog_pytorch import DeepVOG_pytorch
from args import parse_args
from torch.utils.data import DataLoader
from pytorchtools import EarlyStopping, load_from_file
from bdcn_new import BDCN
from test import calc_acc
import pandas as pd
from torchvision.utils import make_grid


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

#%%
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # Deactive file locking
embed_log = 5
EPS=1e-7


torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)
def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return
if __name__ == '__main__':

    args = parse_args()

    device=torch.device("cuda")
    torch.cuda.manual_seed_all(12)

    if torch.cuda.device_count() > 1:
        print('Moving to a multiGPU setup.')
        args.useMultiGPU = True
    else:
        print('Single GPU setup')
        args.useMultiGPU = False


    # load model
    BDCN_network = BDCN()

    state_dict = torch.load('GMM_gen.pt')
    #state_dict = torch.load('gen_00000016.pt')
    BDCN_network.load_state_dict(state_dict['a'])
    BDCN_network = BDCN_network.cuda()
    BDCN_network.eval()

    # print('!!!!!-----MODEL STRUCTURE------------')
    # tot_parameters = 0
    # for k, v in model.named_parameters():
    #     print(k, v.shape)
    #     tot_parameters += v.numel()
    # print('tot_para: ', tot_parameters)
    # print('------------------------------------\n\n\n')

    # calc score via loop
    if(args.id == 0): # 3090
        model_list = [#'logs/ritnet_v2/baseline_edge_2/weights/ritnet_v2_19.pkl',
                      #'logs/deepvog_49.pkl',
                      'logs/ritnet_v1_21.pkl',
                      'logs/baseline_37.pkl',      # baseline
                      'logs/baseline_edge_16.pkl',  # baseline_edge
                      #'logs/baseline_onlyedge_26.pkl',           # baseline_only_aug
                      #'logs/ritnet_v2/baseline_edge/weights/ritnet_v2_49.pkl', # baseline + Edge(img + edge)
                      #'logs/ritnet_v2/baseline_edge_2/weights/ritnet_v2_49.pkl'
        ]# baseline + Edge(img + edge)

        setting_list = [#'baseline_edge.yaml',
                        #'baseline.yaml',
                        'baseline.yaml',
                        'baseline.yaml',
                        'baseline_edge.yaml',
                        #'baseline_only_edge.yaml',
                        #'baseline_edge.yaml',
                        #'baseline_edge.yaml'
        ]

        m_list = [
            #'deepvog',
            'ritnet',
            'ritnet_v2',
            'ritnet_v2',
        ]
    elif args.id == 1: # yunnao
        model_list = ['logs/ritnet_v2/baseline_48_danka_2/weights/ritnet_v2_37.pkl', # baseline
                      'logs/ritnet_v2/baseline_edge_2/weights/ritnet_v2_19.pkl', # baseline + Edge(img + edge)
                      'logs/ritnet_v2/baseline_onlyedge/weights/ritnet_v2_35.pkl',  # only edge
                      'logs/ritnet_v2/baseline_test_edge/weights/ritnet_v2_32.pkl'] # only edge

        setting_list = ['baseline.yaml',
                        'baseline_edge.yaml',
                        'baseline_only_edge.yaml',
                        'baseline_only_edge.yaml']


    assert len(model_list) == len(setting_list), (len(model_list), len(setting_list))

    dataset_list = ['cond_LPW_hist.pkl',
                    'cond_NVGaze_hist.pkl',
                    'cond_OpenEDS_hist.pkl',
                    'cond_Fuhl_hist.pkl']
    if(args.id == 0):
        path2data_list = ['Datasets/Histogram',
                          'Datasets/Histogram',
                          'Datasets/Histogram',
                          'Datasets/Histogram']
    elif(args.id == 1):
        path2data_list = ['userdata/lpw.zip/TEyeD-h5-Edges',
                          'userdata/lpw.zip/TEyeD-h5-Edges',
                          'userdata/lpw.zip/TEyeD-h5-Edges',
                          'userdata/lpw.zip/TEyeD-h5-Edges']
    else:assert(1 == 2), 'illeagel id'
    assert len(dataset_list) == len(path2data_list)
    # for loop -> testloaders
    for ct, dst_dir in enumerate(dataset_list):
        args.curObj = dst_dir[5:8]
        print('args.curObj change to ', args.curObj, '.....')
        path2data = path2data_list[ct]
        f = open(os.path.join('curObjects', dst_dir), 'rb')
        _, _, testObj = pickle.load(f)
        testObj.augFlag = False
        testObj.path2data = os.path.join(args.path2data, path2data)
        testloader = DataLoader(testObj,
                                 batch_size=args.batchsize,
                                 shuffle=False,
                                 num_workers=args.workers,
                                 drop_last=True)
        score = []
        # for loop -> models
        models_dir = []
        for id, model_dir in enumerate(model_list):
            args.model = m_list[id]
            if(m_list[id] == 'ritnet_v2'):
                args.method = setting_list[id][9:-5]
            else: args.method = m_list[id]
            print('args.method change to ', args.curObj, '.....')
            models_dir.append(model_dir)
            setting = get_config('configs/' + setting_list[id])
            if (m_list[id] == 'ritnet_v2'):
                model = DenseNet2D(setting)
            elif (m_list[id] == 'deepvog'):
                model = DeepVOG_pytorch()
            elif(m_list[id] == 'ritnet'):
                model = RITNet()
            else:
                assert 1 == 2, 'illegal model'

            netDict = load_from_file([model_dir])
            model.load_state_dict(netDict['state_dict'])
            # print('!!!!!-----MODEL STRUCTURE------------')
            # tot_parameters = 0
            # for k, v in model.named_parameters():
            #     print(k, v.shape)
            #     tot_parameters += v.numel()
            # print('tot_para: ', tot_parameters)
            # print('------------------------------------\n\n\n')

            model = model.to(device).to(args.prec)  # NOTE: good habit to do this before optimizer
            model = model if not args.useMultiGPU else torch.nn.DataParallel(model)
            print('!!!!!!! ', dst_dir, ' / ', model_dir)
            calc_acc(args, testloader, model, BDCN_network, device, True)
            continue
            ious, reg_pup, reg_iris, seg_pup, seg_iris = calc_acc(args, testloader, model, BDCN_network, device, True)
            continue
            score.append((np.mean(ious), ious[0], ious[1], ious[2], reg_pup, reg_iris, seg_pup, seg_iris))
        print(np.array(score))
        continue
        data = np.array(score).reshape(-1, 8)
        data_df = pd.DataFrame(data)
        data_df.columns = ['mIou', 'bg_iou', 'iris_iou', 'pup_iou', 'reg_pup', 'reg_iris', 'seg_pup', 'seg_iris']
        data_df.index = models_dir
        writer = pd.ExcelWriter('model_score/' + time.strftime("%y%m%d_%H_%M") + dst_dir[5:-4] + '.xlsx')
        data_df.to_excel(writer, float_format='%.3f')
        writer.save()










