#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 03:34:22 2021

@author: rakshit
"""

import os, pickle
import sys
import cv2
import copy
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from loss import get_seg2ptLoss

from tqdm import tqdm
from pathlib import Path
from pprint import pprint

from utils import get_predictions, search_proper_parameter_iou_for_our_data
from models.RITnet_v2 import DenseNet2D
from models.deepvog_pytorch import DeepVOG_pytorch
from helperfunctions import plot_segmap_ellpreds, getValidPoints
from helperfunctions import ransac, ElliFit, my_ellipse, mypause
from pytorchtools import EarlyStopping, load_from_file
from bdcn_new import BDCN
from test import generateImageGrid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path2data', type=str, default='/media/rakshit/Monster/Datasets/Gaze-in-Wild',
                        help='path to eye videos')
    parser.add_argument('--model', type=str, default='ritnet_v2',
                        help='model to load')
    parser.add_argument('--prec', type=int, default=32, help='precision. 16, 32, 64')
    parser.add_argument('--save_maps', type=int, default=0,
                        help='save segmentation maps')
    parser.add_argument('--save_overlay', type=int, default=1,
                        help='save output overlay')
    parser.add_argument('--vid_ext', type=str, default='mp4',
                        help='process videos with given extension')
    parser.add_argument('--loadfile', type=str, default='./weights/all.git_ok',
                        help='choose the weights you want to evalute the videos with. Recommended: all')
    parser.add_argument('--align_width', type=int, default=1,
                        help='reshape videos by matching width, default: True')
    parser.add_argument('--eval_on_cpu', type=int, default=0,
                        help='evaluate using CPU instead of GPU')
    parser.add_argument('--check_for_string_in_fname', type=str, default='',
                        help='process video with a certain string in filename')
    parser.add_argument('--ellseg_ellipses', type=int, default=1,
                        help='use ellseg proposed ellipses, if FALSE, it will fit an ellipse to segmentation mask')
    parser.add_argument('--skip_ransac', type=int, default=0,
                        help='if using ElliFit, it skips outlier removal')
    parser.add_argument('--method', type=str, default='baseline', help=' 0 -> iris, 1 -> pupil')

    args = parser.parse_args()
    opt = vars(args)
    print('------')
    print('parsed arguments:')
    pprint(opt)
    return args

#%% Preprocessing functions and module
# Input frames must be resized to 320X240
def preprocess_frame(img, op_shape, align_width=True):
    if align_width:
        if op_shape[1] != img.shape[1]:
            sc = op_shape[1]/img.shape[1]
            width = int(img.shape[1] * sc)
            height = int(img.shape[0] * sc)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)

            if op_shape[0] > img.shape[0]:
                # Vertically pad array
                pad_width = op_shape[0] - img.shape[0]
                if pad_width%2 == 0:
                    img = np.pad(img, ((pad_width//2, pad_width//2), (0, 0)))
                else:
                    img = np.pad(img, ((np.floor(pad_width/2), np.ceil(pad_width/2)), (0, 0)))
                scale_shift = (sc, pad_width)

            elif op_shape[0] < img.shape[0]:
                # Vertically chop array off
                pad_width = op_shape[0] - img.shape[0]
                if pad_width%2 == 0:
                    img = img[-pad_width/2:+pad_width/2, ...]
                else:
                    img = img[-np.floor(pad_width/2):+np.ceil(pad_width/2), ...]
                scale_shift = (sc, pad_width)

            else:
                scale_shift = (sc, 0)
        else:
            scale_shift = (1, 0)
    else:
        sys.exit('Height alignment not implemented! Exiting ...')

    img = (img - img.mean())/img.std()
    img = torch.from_numpy(img).unsqueeze(0).to(torch.float32) # Add a dummy color channel
    return img, scale_shift
def calc_edge(img, edge_model):
    img_edge = edge_model(torch.cat((img, img, img), dim=1))[-1]
    # if (args.edge_thres == 1):
    #     rt = torch.ones(img_edge.size()).to(device).to(args.prec)
    #     img_edge = torch.where(img_edge >= 0.1, rt, img_edge)
    return img_edge
#%% Forward operation on network
def evaluate_ellseg_on_image(frame, model, edge_model, device=torch.device("cuda")):

    assert len(frame.shape) == 4, 'Frame must be [1,1,H,W]'
    frame_edge = calc_edge(frame, edge_model)
    _, _, H, W = frame.shape
    with torch.no_grad():
        labels = torch.zeros((1, H, W))
        labels[..., 0, 2] = 1
        labels[..., 2, 2] = 2
        op_tup = model(frame.to(device).to(args.prec),
                       frame_edge.to(device).to(args.prec),
                       labels.to(device).long(),
                       torch.zeros((1, 2)).to(device).to(args.prec),
                       torch.zeros((1, 2, 5)).to(device).to(args.prec),
                       torch.zeros((1, H, W)).to(device).to(args.prec),
                       torch.zeros((1, 3, H, W)).to(device).to(args.prec),
                       torch.zeros((1, 4)).to(device).to(args.prec),
                       0,
                       0)
        output, elPred, _, _, elll = op_tup
        seg_out, elPred = output.cpu(), elPred.squeeze().cpu()

        #seg_map = get_predictions(seg_out).squeeze().numpy()
        seg_map = get_predictions(seg_out).squeeze()
        #print(seg_map.shape)

        norm_pupil_ellipse = elPred[5:10]
        norm_iris_ellipse = elPred[0:5]

        # Transformation function H
        _, _, H, W = frame.shape
        H = np.array([[W / 2, 0, W / 2], [0, H / 2, H / 2], [0, 0, 1]])

        pupil_ellipse = my_ellipse(norm_pupil_ellipse.numpy()).transform(H)[0][:-1]
        iris_ellipse = my_ellipse(norm_iris_ellipse.numpy()).transform(H)[0][:-1]

        iris_ellipse = search_proper_parameter_iou_for_our_data((seg_map == 1).cuda(),
                                                       iris_ellipse)
        pupil_ellipse = search_proper_parameter_iou_for_our_data((seg_map == 2).cuda(),
                                                        pupil_ellipse)


    #print(frame.shape, seg_map.shape, elPred.shape)
    # dispI = generateImageGrid(frame.cpu().numpy().squeeze(0),
    #                           seg_map[np.newaxis, :],
    #                           seg_out.cpu().numpy(),
    #                           elPred.detach().cpu().numpy().reshape(-1, 2, 5),
    #                           torch.zeros((1, 2)).numpy(),
    #                           torch.zeros((1, 4)).numpy(),
    #                           elPred.cpu().numpy().reshape(-1, 2, 5),
    #                           # new_para,
    #                           override=True,
    #                           heatmaps=False)

    return frame_edge.detach().cpu().squeeze().numpy(), seg_map.numpy(), pupil_ellipse, iris_ellipse

#%% Rescale operation to bring segmap, pupil and iris ellipses back to original res
def rescale_to_original(edge_map, seg_map, pupil_ellipse, iris_ellipse, scale_shift, orig_shape):

    # Fix pupil ellipse
    pupil_ellipse[1] = pupil_ellipse[1] - np.floor(scale_shift[1]//2)
    pupil_ellipse[:-1] = pupil_ellipse[:-1]*(1/scale_shift[0])

    # Fix iris ellipse
    iris_ellipse[1] = iris_ellipse[1] - np.floor(scale_shift[1]//2)
    iris_ellipse[:-1] = iris_ellipse[:-1]*(1/scale_shift[0])

    if scale_shift[1] < 0:
        # Pad background
        seg_map = np.pad(seg_map, ((-scale_shift[1]//2, -scale_shift[1]//2), (0, 0)))
        edge_map = np.pad(edge_map, ((-scale_shift[1] // 2, -scale_shift[1] // 2), (0, 0)))
    elif scale_shift[1] > 0:
        # Remove extra pixels
        seg_map = seg_map[scale_shift[1]//2:-scale_shift[1]//2, ...]
        seg_map = np.pad(seg_map, ((-scale_shift[1] // 2, -scale_shift[1] // 2), (0, 0)))
        edge_map = edge_map[scale_shift[1] // 2:-scale_shift[1] // 2, ...]
        edge_map = np.pad(edge_map, ((-scale_shift[1] // 2, -scale_shift[1] // 2), (0, 0)))

    seg_map = cv2.resize(seg_map, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
    edge_map = cv2.resize(edge_map, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
    return edge_map, seg_map, pupil_ellipse, iris_ellipse

#%% Definition for processing per video
def evaluate_ellseg_per_video(path_vid, args, model, edge_model):
    path_dir, full_file_name = os.path.split(path_vid)
    print('evaluate {}...'.format(full_file_name))
    file_name = os.path.splitext(full_file_name)[0]

    if args.eval_on_cpu:
        device=torch.device("cpu")
    else:
        device=torch.device("cuda")

    if args.check_for_string_in_fname in file_name:
        print('Processing file: {}'.format(path_vid))
    else:
        print('Skipping video {}'.format(path_vid))
        return False

    vid_obj = cv2.VideoCapture(str(path_vid))

    FR_COUNT = vid_obj.get(cv2.CAP_PROP_FRAME_COUNT)
    FR = vid_obj.get(cv2.CAP_PROP_FPS)
    H  = vid_obj.get(cv2.CAP_PROP_FRAME_HEIGHT)
    W  = vid_obj.get(cv2.CAP_PROP_FRAME_WIDTH)

    path_vid_out = os.path.join(path_dir, file_name+'_ellseg_fit_' + args.method + '.mp4')
    edge_filename = os.path.join(path_dir, file_name+'_edge_' + args.method + '.mp4')
    # if(os.path.exists(path_vid_out)):
    #     return 0
    print('!!!generate {}...'.format(path_vid_out))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid_out = cv2.VideoWriter(path_vid_out, fourcc, int(FR), (int(W), int(H)))
    edge_out = cv2.VideoWriter(edge_filename, fourcc, int(FR), (int(W), int(H)))
    # Dictionary to save output ellipses
    ellipse_out_dict = {}

    ret = True
    pbar = tqdm(total=FR_COUNT)

    counter = 0
    j = 0
    app_center = [[], []]
    while ret:
        ret, ori_frame = vid_obj.read()
        copy_frame = copy.deepcopy(ori_frame)
        if ret == False:
            continue
        tmp_an = []
        eye_splited_img = [0, 1]
        for i in range(2):
            eye_splited_img[i] = ori_frame[:, (0 + i * 320): (320 + i * 320), :]

            frame = cv2.cvtColor(eye_splited_img[i], cv2.COLOR_BGR2GRAY)

            frame_scaled_shifted, scale_shift = preprocess_frame(frame, (240, 320), args.align_width)

            input_tensor = frame_scaled_shifted.unsqueeze(0).to(device)

            # Run the prediction network
            edge_map, seg_map, pupil_ellipse, iris_ellipse = evaluate_ellseg_on_image(input_tensor, model, edge_model)
           # print('!!!{} {} :'.format(j, i), pupil_ellipse)
            app_center[i].append((pupil_ellipse[0], pupil_ellipse[1]))
            # Return ellipse predictions back to original dimensions
            # if(i == 1):
            #     if j == 0:
            #         # print('test2', disp.shape)
            #         # print('test ', disp.permute(1, 2, 0).shape)
            #         h_im = plt.imshow(disp.permute(1, 2, 0))
            #         plt.pause(1)
            #     else:
            #         h_im.set_data(disp.permute(1, 2, 0))
            #         mypause(1)
            # print(' seg_map : ', seg_map.shape)
            # print('edge_map : ', edge_map.shape)
            edge_map *= 255
            edge_map = 255 - edge_map
            edge_map, seg_map, pupil_ellipse, iris_ellipse = rescale_to_original(edge_map, seg_map, pupil_ellipse, iris_ellipse,
                                                                       scale_shift, frame.shape)
            tmp_an.extend(pupil_ellipse)
            tmp_an.extend(iris_ellipse)
            ellipse_out_dict[j] = (iris_ellipse, pupil_ellipse)
            # Generate visuals
            frame_overlayed_with_op = plot_segmap_ellpreds(frame, seg_map, pupil_ellipse, iris_ellipse)

            ori_frame[:, (0 + i * 320): (320 + i * 320), :] = frame_overlayed_with_op
            edge_map_stack = np.stack([edge_map for i in range(0, 3)], axis=2)
           # print('before edge_map_stack.shape : ', edge_map_stack.shape)
            #edge_map_stack = np.moveaxis(edge_map_stack, 0, 3)
            #print('edge_map_stack.shape : ', edge_map_stack.shape)
            copy_frame[:, (0 + i * 320): (320 + i * 320), :] = edge_map_stack
        #datas.append(tmp_an)
        cv2.putText(ori_frame, str(j), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
        vid_out.write(ori_frame)
        edge_out.write(copy_frame)
        j += 1
        pbar.update(1)
        counter += 1
    filename = 'our_data_test/app_centers.pkl'
    print('pickle.dump : {} itmes..'.format(len(app_center[0])))
    f = open(filename, 'wb')
    pickle.dump(app_center, f)
    f.close()

    vid_out.release()
    edge_out.release()
    vid_obj.release()
    pbar.close()

    # Save out ellipse dictionary
    np.save(os.path.join(path_dir, file_name+'_pred2_' + args.method + '.npy'), ellipse_out_dict)
    print('Save ellipse_out_dict to {}....'.format(file_name+'_pred2_' + args.method + '.npy'))

    return True

import yaml
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)
if __name__=='__main__':
    args = parse_args()
    args.prec = torch.float32
    # Load Edge Extract Network
    BDCN_network = BDCN()

    #state_dict = torch.load('GMM_gen.pt')
    print('!!!!!!!!!! gen_00000030_20210610.pt')
    state_dict = torch.load('gen_00000030_20210610.pt')
    print('Attention!!!!!!! this edge model is trained in real dataset....')
    BDCN_network.load_state_dict(state_dict['a'])
    BDCN_network = BDCN_network.cuda()
    BDCN_network.eval()
    # img = cv2.imread("our_data_test/matched.png")
    # frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # frame_scaled_shifted, scale_shift = preprocess_frame(frame, (240, 320), args.align_width)
    #
    # input_tensor = frame_scaled_shifted.unsqueeze(0).to(torch.device("cuda"))
    # input_edge = calc_edge(input_tensor, BDCN_network)
    # print(input_edge.shape)
    # I = input_edge[0][0].detach().cpu().numpy().squeeze()
    # I *= 255
    # I = 255 - I
    # print(I.shape)
    # # cv2.imwrite(os.path.join('img', args.curObj, args.visual_dir,
    # #                          str(id) + '_edge_' + args.method + '.jpg'), I)
    # cv2.imwrite(os.path.join('our_data_test', 'img_edge2.png'), I)
    # exit(0)


    #%% Load network, weights and get ready to evalute
    # # DeepVOG
    # model_file = ''
    # args.method = 'deepvog'
    # # RITNet
    # model_file = ''
    # args.method = 'ritnet'

    # EllSeg
    # model_file = 'logs/baseline_37.pkl'
    # model_file = 'weights/al1.git_ok'
    # setting_file = 'configs/baseline.yaml'
    # args.method = 'baseline'

    # # edge
    model_file = 'logs/baseline_edge_16.pkl'
    setting_file = 'configs/baseline_edge.yaml'
    args.method = ''

    setting = get_config(setting_file)
    if (args.model == 'ritnet_v2'):
        model = DenseNet2D(setting)
    elif (args.model == 'deepvog'):
        model = DeepVOG_pytorch()
    else:
        assert 1 == 2, 'illegal model'
    print('Setting : {} {}'.format(model_file, setting_file))
    netDict = load_from_file([model_file])
    model.load_state_dict(netDict['state_dict'])
    if not args.eval_on_cpu:
        model.cuda()

    #%% Read in each video
    path_obj = Path(args.path2data).rglob('*.avi')

    for path_vid in path_obj:
        if '_ellseg' not in str(path_vid):
            evaluate_ellseg_per_video(path_vid, args, model, BDCN_network)

