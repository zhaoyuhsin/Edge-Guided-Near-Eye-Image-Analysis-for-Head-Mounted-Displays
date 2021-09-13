#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:17:32 2020

@author: rakshit
"""

# This file contains definitions which are not applicable in regular scenarios.
# For general purposes functions, classes and operations - please use helperfunctions.
import os
import cv2
import tqdm
import copy
import torch, time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import make_grid
from skimage import draw
from typing import Optional
from sklearn import metrics
from helperfunctions import my_ellipse
from calc_box_iou import calc_ell_bbox_iou
from sklearn.metrics import jaccard_score
def create_meshgrid(
        height: int,
        width: int,
        normalized_coordinates: Optional[bool] = True) -> torch.Tensor:
    """Generates a coordinate grid for an image.

    When the flag `normalized_coordinates` is set to True, the grid is
    normalized to be in the range [-1,1] to be consistent with the pytorch
    function grid_sample.
    http://pytorch.org/docs/master/nn.html#torch.nn.functional.grid_sample

    Args:
        height (int): the image height (rows).
        width (int): the image width (cols).
        normalized_coordinates (Optional[bool]): whether to normalize
          coordinates in the range [-1, 1] in order to be consistent with the
          PyTorch function grid_sample.

    Return:
        torch.Tensor: returns a grid tensor with shape :math:`(1, H, W, 2)`.
    """
    # generate coordinates
    xs: Optional[torch.Tensor] = None
    ys: Optional[torch.Tensor] = None
    if normalized_coordinates:
        xs = torch.linspace(-1, 1, width)
        ys = torch.linspace(-1, 1, height)
    else:
        xs = torch.linspace(0, width - 1, width)
        ys = torch.linspace(0, height - 1, height)
    # generate grid by stacking coordinates
    base_grid: torch.Tensor = torch.stack(
        torch.meshgrid([xs, ys])).transpose(1, 2)  # 2xHxW
    return torch.unsqueeze(base_grid, dim=0).permute(0, 2, 3, 1)  # 1xHxWx2

def get_nparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_predictions(output):
    '''
    Parameters
    ----------
    output : torch.tensor
        [B, C, *] tensor. Returns the argmax for one-hot encodings.

    Returns
    -------
    indices : torch.tensor
        [B, *] tensor.

    '''
    bs,c,h,w = output.size()
    values, indices = output.cpu().max(1)
    indices = indices.view(bs,h,w) # bs x h x w
    return indices

class Logger():
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        self.dirname = dirname
        self.log_file = open(output_name, 'a+')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write_silent(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()

    def write(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        print (msg)
    def write_summary(self,msg):
        self.log_file.write(msg)
        self.log_file.write('\n')
        self.log_file.flush()
        print (msg)

def getSeg_metrics(y_true, y_pred, cond):
    '''
    Iterate over each batch and identify which classes are present. If no
    class is present, i.e. all 0, then ignore that score from the average.
    Note: This function computes the nan mean. This is because datasets may not
    have all classes present.
    '''
    assert y_pred.ndim==3, 'Incorrect number of dimensions'
    assert y_true.ndim==3, 'Incorrect number of dimensions'

    cond = cond.astype(np.bool)
    B = y_true.shape[0]
    score_list = []
    for i in range(0, B):
        labels_present = np.unique(y_true[i, ...])
        score_vals = np.empty((3, ))
        score_vals[:] = np.nan
        if not cond[i]:
            score = metrics.jaccard_score(y_true[i, ...].reshape(-1),
                                          y_pred[i, ...].reshape(-1),
                                          labels=labels_present,
                                          average=None)
            # Assign score to relevant location
            for j, val in np.ndenumerate(labels_present):
                score_vals[val] = score[j]
        score_list.append(score_vals)
    score_list = np.stack(score_list, axis=0)
    score_list_clean = score_list[~cond, :] # Only select valid entries
    perClassIOU = np.nanmean(score_list_clean, axis=0) if len(score_list_clean) > 0 else np.nan*np.ones(3, )
    meanIOU = np.nanmean(perClassIOU) if len(score_list_clean) > 0 else np.nan
    return meanIOU, perClassIOU, score_list

def getPoint_metric(y_true, y_pred, cond, sz, do_unnorm):
    # Unnormalize predicted points
    if do_unnorm:
        y_pred = unnormPts(y_pred, sz)

    cond = cond.astype(np.bool)
    flag = (~cond).astype(np.float)
    dist = metrics.pairwise_distances(y_true, y_pred, metric='euclidean')
    dist = flag*np.diag(dist)
    return (np.sum(dist)/np.sum(flag) if np.any(flag) else np.nan,
            dist)

def getAng_metric(y_true, y_pred, cond):
    # Assumes the incoming angular measurements are in radians
    cond = cond.astype(np.bool)
    flag = (~cond).astype(np.float)
    dist = np.rad2deg(flag*np.abs(y_true - y_pred))
    return (np.sum(dist)/np.sum(flag) if np.any(flag) else np.nan,
            dist)

def pri(ellipse):
    angle = ellipse[4] * 180 / 3.14149
    print(ellipse[4])
    return "{:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(ellipse[0], ellipse[1], ellipse[2], ellipse[3], angle)
def calc_ell_iou(seg, el_iris, mesh, nor = True, angle_nor = False):
    #st_time = time.time()
    # seg : [W, H]
    # el_iris : [6]
    # print('parameter : ',el_iris)
    if(angle_nor):
        el_iris[4] = el_iris[4] / 180. * 3.14159
    if(not nor):
        #print('Before : ', el_iris)
        H = np.array([[2 / seg.shape[1], 0, -1], [0, 2 / seg.shape[0], -1], [0, 0, 1]])
        el_iris = my_ellipse(el_iris).transform(H)[0][:-1]
        #print(el_iris)
    #now1 = time.time()
    ell_map = np.zeros(seg.shape)
    X = (mesh[..., 0].squeeze() - el_iris[0]) * np.cos(el_iris[-1]) + \
        (mesh[..., 1].squeeze() - el_iris[1]) * np.sin(el_iris[-1])
    Y = -(mesh[..., 0].squeeze() - el_iris[0]) * np.sin(el_iris[-1]) + \
        (mesh[..., 1].squeeze() - el_iris[1]) * np.cos(el_iris[-1])
    wtMat = (X / el_iris[2]) ** 2 + (Y / el_iris[3]) ** 2 - 1
    [rr_i, cc_i] = np.where(wtMat <= 0)
    ell_map[rr_i, cc_i] = 1

    #now2 = time.time()
    ell_map = torch.Tensor(ell_map).cuda()
    c = seg.mul(ell_map)
    score = torch.sum(c) / (torch.sum(seg) + torch.sum(ell_map) - torch.sum(c))
    #now3 = time.time()
    #print(now1 - st_time, now2 - now1, now3 - now2)
    return score.item()

def generateImageGrid(I,
                      mask,
                      #mask_prob,
                      elNorm,
                      pupil_center,
                      cond,
                      elNorm_gt,
                      heatmaps=False,
                      override=False):
    #print('generateImageGrid() : ', I.shape, mask.shape, mask_prob.shape, elNorm.shape)
    '''
    Parameters
    ----------
    I : numpy array [B, H, W]
        A batchfirst array which holds images
    mask : numpy array [B, H, W]
        A batch first array which holds for individual pixels.
    hMaps: numpy array [B, C, N, H, W]
        N is the # of points, C is the category the points belong to (iris or
        pupil). Heatmaps are gaussians centered around point of interest
    elNorm:numpy array [B, C, 5]
        Normalized ellipse parameters
    pupil_center : numpy array [B, 2]
        Identified pupil center for plotting.
    cond : numpy array [B, 5]
        A flag array which holds information about what information is present.
    heatmaps : bool, optional
        Unless specificed, does not show the heatmaps of predicted points
    override : bool, optional
        An override flag which plots data despite being demarked in the flag
        array. Generally used during testing.
        The default is False.

    Returns
    -------
    I_o : numpy array [Ho, Wo]
        Returns an array holding concatenated images from the input overlayed
        with segmentation mask, pupil center and pupil ellipse.

    Note: If masks exist, then ellipse parameters would exist aswell.
    '''
    B, H, W = I.shape
    mesh = create_meshgrid(H, W, normalized_coordinates=True) # 1xHxWx2
    H = np.array([[W/2, 0, W/2], [0, H/2, H/2], [0, 0, 1]])
    I_o = []
    # for i in range(0, min(4, cond.shape[0])):
    #     im = mask_prob[i][2]
    #     print(im)
    #     im *= 255
    #     ze = np.zeros(im.shape)
    #     im = np.stack([im, ze, ze], axis=2)
    #     I_o.append(im)
    # for i in range(0, min(4, cond.shape[0])):
    #     im = mask_prob[i][1]
    #     im *= 255
    #     #im = 255 - im
    #     im = np.stack([im, ze, ze], axis=2)
    #     I_o.append(im)
    # for i in range(0, min(16, cond.shape[0])):
    #     gtp_score = calc_ell_iou(mask[i] == 2, elNorm_gt[i, 1, ...], mesh)
    #     p_score = calc_ell_iou(mask[i] == 2, elNorm[i, 1, ...], mesh)
    #     gti_socre = calc_ell_iou(mask[i] == 1, elNorm_gt[i, 0, ...], mesh)
    #     gti_score = calc_ell_iou(mask[i] == 1, elNorm[i, 0, ...], mesh)
    for i in range(0, min(4, cond.shape[0])):
        im = I[i, ...].squeeze() - I[i, ...].min()
        #im = np.uint8(255*im/im.max())
        im = cv2.equalizeHist(np.uint8(255*im/im.max()))
        im = im - im.min()
        im = im/im.max()

        im = np.stack([im for i in range(0, 3)], axis=2)

        if (not cond[i, 1]) or override:
            # If masks exists

            rr, cc = np.where(mask[i, ...] == 1)
            im[rr, cc, ...] = np.array([0, 255, 0]) # Green
            rr, cc = np.where(mask[i, ...] == 2)
            im[rr, cc, ...] = np.array([255, 255, 0]) # Yellow


            # Just for experiments. Please ignore.
            el_iris = elNorm[i, 0, ...]
            X = (mesh[..., 0].squeeze() - el_iris[0])*np.cos(el_iris[-1])+\
                (mesh[..., 1].squeeze() - el_iris[1])*np.sin(el_iris[-1])
            Y = -(mesh[..., 0].squeeze() - el_iris[0])*np.sin(el_iris[-1])+\
                 (mesh[..., 1].squeeze() - el_iris[1])*np.cos(el_iris[-1])
            wtMat = (X/el_iris[2])**2 + (Y/el_iris[3])**2 - 1
            # [rr_i, cc_i] = np.where(wtMat< 0)

            try:
                # print('elNorm : ', elNorm[i])
                # print('elNorm_GT : ', elNorm_gt[i])
                el_iris = my_ellipse(elNorm[i, 0, ...]).transform(H)[0]
                el_pupil = my_ellipse(elNorm[i, 1, ...]).transform(H)[0]
                el_iris_gt = my_ellipse(elNorm_gt[i, 0, ...]).transform(H)[0]
                el_pupil_gt = my_ellipse(elNorm_gt[i, 1, ...]).transform(H)[0]
                # print(el_iris)
                # print(el_iris_gt)
                # print(el_pupil)
                # print(el_pupil_gt)


            except:
                print('Warning: inappropriate ellipses. Defaulting to not break runtime..')
                el_iris = np.array([W/2, H/2, W/8, H/8, 0.0]).astype(np.float32)
                el_pupil = np.array([W/2, H/2, W/4, H/4, 0.0]).astype(np.float32)
            # iris_bbiou = calc_ell_bbox_iou(el_iris[:5], el_iris_gt[:5])
            # pupil_bbiou = calc_ell_bbox_iou(el_pupil[:5], el_pupil_gt[:5])

            [rr_i_gt, cc_i_gt] = draw.ellipse_perimeter(int(el_iris_gt[1]),
                                              int(el_iris_gt[0]),
                                              int(el_iris_gt[3]),
                                              int(el_iris_gt[2]),
                                              orientation=el_iris_gt[4])
            [rr_p_gt, cc_p_gt] = draw.ellipse_perimeter(int(el_pupil_gt[1]),
                                             int(el_pupil_gt[0]),
                                             int(el_pupil_gt[3]),
                                             int(el_pupil_gt[2]),
                                             orientation=el_pupil_gt[4])
            rr_i_gt = rr_i_gt.clip(6, im.shape[0]-6)
            rr_p_gt = rr_p_gt.clip(6, im.shape[0]-6)
            cc_i_gt = cc_i_gt.clip(6, im.shape[1]-6)
            cc_p_gt = cc_p_gt.clip(6, im.shape[1]-6)

            # im[rr_i_gt, cc_i_gt, ...] = np.array([0, 102, 255])
            # im[rr_p_gt, cc_p_gt, ...] = np.array([0, 102, 255])

            [rr_i, cc_i] = draw.ellipse_perimeter(int(el_iris[1]),
                                                  int(el_iris[0]),
                                                  int(el_iris[3]),
                                                  int(el_iris[2]),
                                                  orientation=el_iris[4])
            [rr_p, cc_p] = draw.ellipse_perimeter(int(el_pupil[1]),
                                                  int(el_pupil[0]),
                                                  int(el_pupil[3]),
                                                  int(el_pupil[2]),
                                                  orientation=el_pupil[4])
            rr_i = rr_i.clip(6, im.shape[0] - 6)
            rr_p = rr_p.clip(6, im.shape[0] - 6)
            cc_i = cc_i.clip(6, im.shape[1] - 6)
            cc_p = cc_p.clip(6, im.shape[1] - 6)

            im[rr_i, cc_i, ...] = np.array([0, 0, 255])
            im[rr_p, cc_p, ...] = np.array([255, 0, 0])
            # print(i)
            # print('iris    : ', pri(el_iris))
            # print('iris gt : ', pri(el_iris_gt))
            # print('pupil   : ', pri(el_pupil))
            # print('pupil gt: ', pri(el_pupil_gt))
            # print('iris    : ', pri(elNorm[i][0]))
            # print('iris gt : ', pri(elNorm_gt[i][0]))
            # print('pupil   : ', pri(elNorm[i][1]))
            # print('pupil gt: ', pri(elNorm_gt[i][1]))
            # gtp_score = calc_ell_iou(mask[i] == 2, elNorm_gt[i, 1, ...], mesh)
            # p_score = calc_ell_iou(mask[i] == 2, elNorm[i, 1, ...], mesh)
            # center = [el_pupil[0], el_pupil[1]]
            # ans = [el_pupil[2], el_pupil[3], el_pupil[4]]
            # rt = calc_ell_iou(mask[i] == 2, np.array(center + ans), mesh)
            # for a in range(-10, 11):
            #     for b in range(-10, 11):
            #         for alpha in range(-40, 40):
            #             now = [el_pupil[2] + a, el_pupil[3] + b, el_pupil[4] + alpha]
            #             score = calc_ell_iou(mask[i] == 2, np.array(center + now), mesh)
            #             if(score > rt):
            #                 rt = score
            #                 ans = now
            # print('search successfully.....')
            # print(p_score, gtp_score)
            # print(rt)
            # print(ans)




            # cv2.putText(im, str('GT Pupil IoU:' + str(round((gtp_score), 2))), (10, 30), cv2.FONT_HERSHEY_PLAIN,
            #             2.0, (255, 0, 0), 2)
            # cv2.putText(im, str('Pupil IoU: ' + str(round(p_score, 2))), (10, 60), cv2.FONT_HERSHEY_PLAIN,
            #             2.0, (255, 0, 0), 2)

        # if (not cond[i, 0]) or override:
        #     # If pupil center exists
        #     rr, cc = draw.disk((pupil_center[i, 1].clip(6, im.shape[0]-6),
        #                         pupil_center[i, 0].clip(6, im.shape[1]-6)),
        #                          5)
        #     im[rr, cc, ...] = 255
        I_o.append(im)
    I_o = np.stack(I_o, axis=0)
    I_o = np.moveaxis(I_o, 3, 1)
    I_o = make_grid(torch.from_numpy(I_o).to(torch.float), nrow=4)
    #I_o = I_o - I_o.min()
    #I_o = I_o/I_o.max()
    #exit(0)
    return I_o

def search_proper_parameter_iou(seg, elNorm, elNorm_gt, ell_para, ell_para_gt):
    H, W = seg.shape
    mesh = create_meshgrid(H, W, normalized_coordinates=True)  # 1xHxWx2
    # # search suitable pupil parameters(a, b, alpha)
    gtp_score = calc_ell_iou(seg, elNorm_gt, mesh)
    p_score = calc_ell_iou(seg, elNorm, mesh)
    center = [ell_para[0], ell_para[1]]
    ans = [ell_para[2], ell_para[3], ell_para[4] * 180. / 3.14159]
    rt = calc_ell_iou(seg, np.array(center + ans), mesh, False, True)
    # print('pupil   : ', pri(ell_para))
    # print('pupil gt: ', pri(ell_para_gt))
    # print(rt, ans, gtp_score)
    # print('!!!!!!!----------start--search-------')
    now = ans.copy()
    d = [1., 1., 1.]
    for tt in range(40):
        flag = False
        for j in range(3):
            now[j] -= d[j]
            score = calc_ell_iou(seg, np.array(center + now), mesh, False, True)
            if(score > rt):
                flag = True
                continue
            now[j] += 2. * d[j]
            score = calc_ell_iou(seg, np.array(center + now), mesh, False, True)
            if (score > rt):
                flag = True
                continue
            now[j] -= d[j]
            d[j] *= 0.8 # decrease learning_rate
        score = calc_ell_iou(seg, np.array(center + now), mesh, False, True)
        if (score > rt):
            # print(tt, 'score ', rt, '----> ', score, now)
            rt = score
        if(not flag):break

    # normal inverse
    # print('search successfully.....')
    # print(p_score, gtp_score)
    # print(rt)
    # print(ans)
    # l_pupil_before = np.array(center + ans)
    l_pupil_after = np.array(center + now)
    # l_pupil_before[4] = l_pupil_before[4] / 180.0 * 3.14159
    l_pupil_after[4] = l_pupil_after[4] / 180.0 * 3.14159
    # before_bbiou = calc_ell_bbox_iou(l_pupil_before[:5], ell_para_gt[:5])
    after_bbiou = calc_ell_bbox_iou(l_pupil_after[:5], ell_para_gt[:5])
    # print('!!!!!!! bbox IoU : ', before_bbiou, '-----> ', after_bbiou)
    return after_bbiou, l_pupil_after
def search_proper_parameter_iou_for_our_data(seg, ell_para):
    H, W = seg.shape
    mesh = create_meshgrid(H, W, normalized_coordinates=True)  # 1xHxWx2


    center = [ell_para[0], ell_para[1]]
    ans = [ell_para[2], ell_para[3], ell_para[4] * 180. / 3.14159]
    rt = calc_ell_iou(seg, np.array(center + ans), mesh, False, True)

    now = ans.copy()
    d = [1., 1., 1.]
    for tt in range(40):
        flag = False
        for j in range(3):
            now[j] -= d[j]
            score = calc_ell_iou(seg, np.array(center + now), mesh, False, True)
            if(score > rt):
                flag = True
                continue
            now[j] += 2. * d[j]
            score = calc_ell_iou(seg, np.array(center + now), mesh, False, True)
            if (score > rt):
                flag = True
                continue
            now[j] -= d[j]
            d[j] *= 0.8 # decrease learning_rate
        score = calc_ell_iou(seg, np.array(center + now), mesh, False, True)
        if (score > rt):
            # print(tt, 'score ', rt, '----> ', score, now)
            rt = score
        if(not flag):break


    l_pupil_after = np.array(center + now)
    l_pupil_after[4] = l_pupil_after[4] / 180.0 * 3.14159

    return l_pupil_after

def generateImageGrid2(I,
                      mask,
                      elNorm,
                      pupil_center,
                      cond,
                      elNorm_gt,
                      #new_para,
                      heatmaps=False,
                      override=False):

    B, H, W = I.shape
    mesh = create_meshgrid(H, W, normalized_coordinates=True) # 1xHxWx2
    H = np.array([[W/2, 0, W/2], [0, H/2, H/2], [0, 0, 1]])
    I_o = []
    for i in range(0, min(4, cond.shape[0])):
        im = I[i, ...].squeeze() - I[i, ...].min()
        im = cv2.equalizeHist(np.uint8(255*im/im.max()))
        im = np.stack([im for i in range(0, 3)], axis=2)
        # gti_socre = calc_ell_iou(mask[i] == 1, elNorm_gt[i, 0, ...], mesh)
        # gti_score = calc_ell_iou(mask[i] == 1, elNorm[i, 0, ...], mesh)
        # cv2.putText(im, str('GT Iris IoU:' + str(round((gti_socre), 2))), (10, 30), cv2.FONT_HERSHEY_PLAIN,
        #             2.0, (255, 0, 0), 2)
        # cv2.putText(im, str('Iris IoU: ' + str(round(gti_score, 2))), (10, 60), cv2.FONT_HERSHEY_PLAIN,
        #             2.0, (255, 0, 0), 2)
        I_o.append(im)

    for i in range(0, min(4, cond.shape[0])):
        im = I[i, ...].squeeze() - I[i, ...].min()
        im = cv2.equalizeHist(np.uint8(255*im/im.max()))
        im = np.stack([im for i in range(0, 3)], axis=2)

        if (not cond[i, 1]) or override:
            # If masks exists

            # rr, cc = np.where(mask[i, ...] == 1)
            # im[rr, cc, ...] = np.array([0, 255, 0]) # Green
            rr, cc = np.where(mask[i, ...] == 2)
            im[rr, cc, ...] = np.array([255, 255, 0]) # Yellow[255, 255, 0]

            try:
                # print('elNorm : ', elNorm[i])
                # print('elNorm_GT : ', elNorm_gt[i])
                # print('!!!!!', elNorm_gt[i, 1, ...])
                el_iris = my_ellipse(elNorm[i, 0, ...]).transform(H)[0]
                el_pupil = my_ellipse(elNorm[i, 1, ...]).transform(H)[0]
                el_iris_gt = my_ellipse(elNorm_gt[i, 0, ...]).transform(H)[0]
                el_pupil_gt = my_ellipse(elNorm_gt[i, 1, ...]).transform(H)[0]
                el_pupil_new = new_para[i]
                # print(el_iris)
                # print(el_iris_gt)
                # print(el_pupil)
                # print(el_pupil_gt)


            except:
                print('Warning: inappropriate ellipses. Defaulting to not break runtime..')
                el_iris = np.array([W/2, H/2, W/8, H/8, 0.0]).astype(np.float32)
                el_pupil = np.array([W/2, H/2, W/4, H/4, 0.0]).astype(np.float32)
            # iris_bbiou = calc_ell_bbox_iou(el_iris[:5], el_iris_gt[:5])
            # pupil_bbiou = calc_ell_bbox_iou(el_pupil[:5], el_pupil_gt[:5])

            # !!!!!!!!!!GT
            # [rr_i_gt, cc_i_gt] = draw.ellipse_perimeter(int(el_iris_gt[1]),
            #                                   int(el_iris_gt[0]),
            #                                   int(el_iris_gt[3]),
            #                                   int(el_iris_gt[2]),
            #                                   orientation=el_iris_gt[4])
            [rr_p_gt, cc_p_gt] = draw.ellipse_perimeter(int(el_pupil_gt[1]),
                                             int(el_pupil_gt[0]),
                                             int(el_pupil_gt[3]),
                                             int(el_pupil_gt[2]),
                                             orientation=el_pupil_gt[4])
            # rr_i_gt = rr_i_gt.clip(6, im.shape[0]-6)
            rr_p_gt = rr_p_gt.clip(6, im.shape[0]-6)
            # cc_i_gt = cc_i_gt.clip(6, im.shape[1]-6)
            cc_p_gt = cc_p_gt.clip(6, im.shape[1]-6)

            # im[rr_i_gt, cc_i_gt, ...] = np.array([0, 102, 255])
            im[rr_p_gt, cc_p_gt, ...] = np.array([0, 0, 255])

            # !!!!! PREDICT
            [rr_p, cc_p] = draw.ellipse_perimeter(int(el_pupil[1]),
                                                  int(el_pupil[0]),
                                                  int(el_pupil[3]),
                                                  int(el_pupil[2]),
                                                  orientation=el_pupil[4])
            # rr_i = rr_i.clip(6, im.shape[0] - 6)
            rr_p = rr_p.clip(6, im.shape[0] - 6)
            # cc_i = cc_i.clip(6, im.shape[1] - 6)
            cc_p = cc_p.clip(6, im.shape[1] - 6)

            # im[rr_i, cc_i, ...] = np.array([0, 0, 255])
            im[rr_p, cc_p, ...] = np.array([255, 0, 0])

            # !!!!!!!! NEW
            [rr_new, cc_new] = draw.ellipse_perimeter(int(el_pupil_new[1]),
                                                        int(el_pupil_new[0]),
                                                        int(el_pupil_new[3]),
                                                        int(el_pupil_new[2]),
                                                        orientation=el_pupil_new[4])
            # rr_i_gt = rr_i_gt.clip(6, im.shape[0]-6)
            rr_new = rr_new.clip(6, im.shape[0] - 6)
            # cc_i_gt = cc_i_gt.clip(6, im.shape[1]-6)
            cc_new = cc_new.clip(6, im.shape[1] - 6)

            # im[rr_i_gt, cc_i_gt, ...] = np.array([0, 102, 255])
            # im[rr_new, cc_new, ...] = np.array([0, 255, 0])

            # print(i)
            # print('iris    : ', pri(el_iris))
            # print('iris gt : ', pri(el_iris_gt))
            # print('pupil   : ', pri(el_pupil))
            # print('pupil gt: ', pri(el_pupil_gt))
            # print('iris    : ', pri(elNorm[i][0]))
            # print('iris gt : ', pri(elNorm_gt[i][0]))
            # print('pupil   : ', pri(elNorm[i][1]))
            # print('pupil gt: ', pri(elNorm_gt[i][1]))

            # cv2.putText(im, str('GT Pupil IoU:' + str(round((gtp_score), 2))), (10, 30), cv2.FONT_HERSHEY_PLAIN,
            #             2.0, (255, 0, 0), 2)
            # cv2.putText(im, str('Pupil IoU: ' + str(round(p_score, 2))), (10, 60), cv2.FONT_HERSHEY_PLAIN,
            #             2.0, (255, 0, 0), 2)

        if (not cond[i, 0]) or override:
            # If pupil center exists
            rr, cc = draw.disk((pupil_center[i, 1].clip(6, im.shape[0]-6),
                                pupil_center[i, 0].clip(6, im.shape[1]-6)),
                                 5)
            im[rr, cc, ...] = 255
        I_o.append(im)

    I_o = np.stack(I_o, axis=0)
    I_o = np.moveaxis(I_o, 3, 1)
    I_o = make_grid(torch.from_numpy(I_o).to(torch.float), nrow=4)
    I_o = I_o - I_o.min()
    I_o = I_o/I_o.max()
    #exit(0)
    return I_o

def normPts(pts, sz):
    pts_o = copy.deepcopy(pts)
    res = pts_o.shape
    pts_o = pts_o.reshape(-1, 2)
    pts_o[:, 0] = 2*(pts_o[:, 0]/sz[1]) - 1
    pts_o[:, 1] = 2*(pts_o[:, 1]/sz[0]) - 1
    pts_o = pts_o.reshape(res)
    return pts_o

def unnormPts(pts, sz):
    pts_o = copy.deepcopy(pts)
    res = pts_o.shape
    pts_o = pts_o.reshape(-1, 2)
    pts_o[:, 0] = 0.5*sz[1]*(pts_o[:, 0] + 1)
    pts_o[:, 1] = 0.5*sz[0]*(pts_o[:, 1] + 1)
    pts_o = pts_o.reshape(res)
    return pts_o

def calc_edge(args, img, edge_model, device):
    with torch.no_grad():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        img_edge = edge_model(torch.cat((img, img, img), dim=1).to(device).to(args.prec))[-1]

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if (args.edge_thres == 1):
        rt = torch.ones(img_edge.size()).to(device).to(args.prec)
        img_edge = torch.where(img_edge >= 0.1, rt, img_edge)
    return img_edge

def lossandaccuracy(args, loader, model, edge_model, alpha, device):
    '''
    A function to compute validation loss and performance

    Parameters
    ----------
    loader : torch loader
        Custom designed loader found in the helper functions.
    model : torch net
        Initialized model which needs to be validated againt loader.
    alpha : Learning rate factor. Refer to RITNet paper for more information.
        constant.

    Returns
    -------
    TYPE
        validation score.

    '''
    epoch_loss = []
    ious = []

    scoreType = {'c_dist':[], 'ang_dist': [], 'sc_rat': []}
    scoreTrack = {'pupil': copy.deepcopy(scoreType),
                  'iris': copy.deepcopy(scoreType)}

    model.eval()
    latent_codes = []
    with torch.no_grad():
        for bt, batchdata in enumerate(tqdm.tqdm(loader)):
            if(args.test_normal and bt > 3):break
            img, labels, spatialWeights, distMap, pupil_center, iris_center, elNorm, cond, imInfo = batchdata
            img_edge = calc_edge(args, img, edge_model, device)
            op_tup = model(img.to(device).to(args.prec),
                            img_edge.to(device).to(args.prec),
                            labels.to(device).long(),
                            pupil_center.to(device).to(args.prec),
                            elNorm.to(device).to(args.prec),
                            spatialWeights.to(device).to(args.prec),
                            distMap.to(device).to(args.prec),
                            cond.to(device).to(args.prec),
                            imInfo[:, 2].to(device).to(torch.long), # Send DS #
                            alpha)

            output, elOut, latent, loss, _ = op_tup
            latent_codes.append(latent.detach().cpu())
            loss = loss.mean() if args.useMultiGPU else loss
            epoch_loss.append(loss.item())

            pred_c_iri = elOut[:, 0:2].detach().cpu().numpy()
            pred_c_pup = elOut[:, 5:7].detach().cpu().numpy()

            # Center distance
            # ptDist_iri = getPoint_metric(iris_center.numpy(),
            #                              pred_c_iri,
            #                              cond[:,0].numpy(),
            #                              img.shape[2:],
            #                              True)[0] # Unnormalizes the points
            # ptDist_pup = getPoint_metric(pupil_center.numpy(),
            #                              pred_c_pup,
            #                              cond[:,0].numpy(),
            #                              img.shape[2:],
            #                              True)[0] # Unnormalizes the points
            #
            # # Angular distance
            # angDist_iri = getAng_metric(elNorm[:, 0, 4].numpy(),
            #                             elOut[:,  4].detach().cpu().numpy(),
            #                             cond[:, 1].numpy())[0]
            # angDist_pup = getAng_metric(elNorm[:, 1, 4].numpy(),
            #                             elOut[:, 9].detach().cpu().numpy(),
            #                             cond[:, 1].numpy())[0]
            #
            # # Scale metric
            # gt_ab = elNorm[:, 0, 2:4]
            # pred_ab = elOut[:, 2:4].cpu().detach()
            # scale_iri = torch.sqrt(torch.sum(gt_ab**2, dim=1)/torch.sum(pred_ab**2, dim=1))
            # scale_iri = torch.sum(scale_iri*(~cond[:,1]).to(torch.float32)).item()
            # gt_ab = elNorm[:, 1, 2:4]
            # pred_ab = elOut[:, 7:9].cpu().detach()
            # scale_pup = torch.sqrt(torch.sum(gt_ab**2, dim=1)/torch.sum(pred_ab**2, dim=1))
            # scale_pup = torch.sum(scale_pup*(~cond[:,1]).to(torch.float32)).item()
            #
            # predict = get_predictions(output)
            # iou = getSeg_metrics(labels.numpy(),
            #                      predict.numpy(),
            #                      cond[:, 1].numpy())[1]
            # ious.append(iou)
            #
            # # Append to score dictionary
            # scoreTrack['iris']['c_dist'].append(ptDist_iri)
            # scoreTrack['iris']['ang_dist'].append(angDist_iri)
            # scoreTrack['iris']['sc_rat'].append(scale_iri)
            # scoreTrack['pupil']['c_dist'].append(ptDist_pup)
            # scoreTrack['pupil']['ang_dist'].append(angDist_pup)
            # scoreTrack['pupil']['sc_rat'].append(scale_pup)
            #
            # ious.append(iou)
    ious = np.stack(ious, axis=0)

    return (np.mean(epoch_loss),
            np.nanmean(ious, 0),
            scoreTrack,
            latent_codes)

def points_to_heatmap(pts, std, res):
    # Given image resolution and variance, generate synthetic Gaussians around
    # points of interest for heat map regression.
    # pts: [B, C, N, 2] Normalized points
    # H: [B, C, N, H, W] Output heatmap
    B, C, N, _ = pts.shape
    pts = unnormPts(pts, res) #
    grid = create_meshgrid(res[0], res[1], normalized_coordinates=False)
    grid = grid.squeeze()
    X = grid[..., 0]
    Y = grid[..., 1]

    X = torch.stack(B*C*N*[X], axis=0).reshape(B, C, N, res[0], res[1])
    X = X - torch.stack(np.prod(res)*[pts[..., 0]], axis=3).reshape(B, C, N, res[0], res[1])

    Y = torch.stack(B*C*N*[Y], axis=0).reshape(B, C, N, res[0], res[1])
    Y = Y - torch.stack(np.prod(res)*[pts[..., 1]], axis=3).reshape(B, C, N, res[0], res[1])

    H = torch.exp(-(X**2 + Y**2)/(2*std**2))
    #H = H/(2*np.pi*std**2) # This makes the summation == 1 per image in a batch
    return H

def ElliFit(coords, mns):
    '''
    Parameters
    ----------
    coords : torch float32 [B, N, 2]
        Predicted points on ellipse periphery
    mns : torch float32 [B, 2]
        Predicted mean of the center points

    Returns
    -------
    PhiOp: The Phi scores associated with ellipse fitting. For more info,
    please refer to ElliFit paper.
    '''
    B = coords.shape[0]

    PhiList = []

    for bt in range(B):
        coords_norm = coords[bt, ...] - mns[bt, ...] # coords_norm: [N, 2]
        N = coords_norm.shape[0]

        x = coords_norm[:, 0]
        y = coords_norm[:, 1]

        X = torch.stack([-x**2, -x*y, x, y, -torch.ones(N, ).cuda()], dim=1)
        Y = y**2

        a = torch.inverse(X.T.matmul(X))
        b = X.T.matmul(Y)
        Phi = a.matmul(b)
        PhiList.append(Phi)
    Phi = torch.stack(PhiList, dim=0)
    return Phi

def spatial_softmax_2d(input: torch.Tensor, temperature: torch.Tensor = torch.tensor(1.0)) -> torch.Tensor:
    r"""Applies the Softmax function over features in each image channel.
    Note that this function behaves differently to `torch.nn.Softmax2d`, which
    instead applies Softmax over features at each spatial location.
    Returns a 2D probability distribution per image channel.
    Arguments:
        input (torch.Tensor): the input tensor.
        temperature (torch.Tensor): factor to apply to input, adjusting the
          "smoothness" of the output distribution. Default is 1.
    Shape:
        - Input: :math:`(B, N, H, W)`
        - Output: :math:`(B, N, H, W)`
    """

    batch_size, channels, height, width = input.shape
    x: torch.Tensor = input.view(batch_size, channels, -1)

    x_soft: torch.Tensor = F.softmax(x * temperature, dim=-1)

    return x_soft.view(batch_size, channels, height, width)

def spatial_softargmax_2d(input: torch.Tensor, normalized_coordinates: bool = True) -> torch.Tensor:
    r"""Computes the 2D soft-argmax of a given input heatmap.
    The input heatmap is assumed to represent a valid spatial probability
    distribution, which can be achieved using
    :class:`~kornia.contrib.dsnt.spatial_softmax_2d`.
    Returns the index of the maximum 2D coordinates of the given heatmap.
    The output order of the coordinates is (x, y).
    Arguments:
        input (torch.Tensor): the input tensor.
        normalized_coordinates (bool): whether to return the
          coordinates normalized in the range of [-1, 1]. Otherwise,
          it will return the coordinates in the range of the input shape.
          Default is True.
    Shape:
        - Input: :math:`(B, N, H, W)`
        - Output: :math:`(B, N, 2)`
    Examples:
        >>> heatmaps = torch.tensor([[[
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 1., 0.]]]])
        >>> coords = spatial_softargmax_2d(heatmaps, False)
        tensor([[[1.0000, 2.0000]]])
    """

    batch_size, channels, height, width = input.shape

    # Create coordinates grid.
    grid: torch.Tensor = create_meshgrid(
        height, width, normalized_coordinates)
    grid = grid.to(device=input.device, dtype=input.dtype)

    pos_x: torch.Tensor = grid[..., 0].reshape(-1)
    pos_y: torch.Tensor = grid[..., 1].reshape(-1)

    input_flat: torch.Tensor = input.view(batch_size, channels, -1)

    # Compute the expectation of the coordinates.
    expected_y: torch.Tensor = torch.sum(pos_y * input_flat, -1, keepdim=True)
    expected_x: torch.Tensor = torch.sum(pos_x * input_flat, -1, keepdim=True)

    output: torch.Tensor = torch.cat([expected_x, expected_y], -1)

    return output.view(batch_size, channels, 2)  # BxNx2

def soft_heaviside(x, sc, mode):
    '''
    Given an input and a scaling factor (default 64), the soft heaviside
    function approximates the behavior of a 0 or 1 operation in a differentiable
    manner. Note the max values in the heaviside function are scaled to 0.9.
    This scaling is for convenience and stability with bCE loss.
    '''
    sc = torch.tensor([sc]).to(torch.float32).to(x.device)
    if mode==1:
        # Original soft-heaviside
        # Try sc = 64
        return 0.9/(1 + torch.exp(-sc/x))
    elif mode==2:
        # Some funky shit but has a nice gradient
        # Try sc = 0.001
        return 0.45*(1 + (2/np.pi)*torch.atan2(x, sc))
    elif mode==3:
        # Good ol' scaled sigmoid. FUTURE: make sc free parameter
        # Try sc = 8
        return torch.sigmoid(sc*x)
    else:
        print('Mode undefined')

def _assert_no_grad(variables):
    for var in variables:
        assert not var.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these variables as volatile or not requiring gradients"

def cdist(x, y):
    '''
    Input: x is a Nxd Tensor
           y is a Mxd Tensor
    Output: dist is a NxM matrix where dist[i,j] is the norm
           between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||
    '''
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences**2, -1).sqrt()
    return distances

def generaliz_mean(tensor, dim, p=-9, keepdim=False):
    # """
    # Computes the softmin along some axes.
    # Softmin is the same as -softmax(-x), i.e,
    # softmin(x) = -log(sum_i(exp(-x_i)))

    # The smoothness of the operator is controlled with k:
    # softmin(x) = -log(sum_i(exp(-k*x_i)))/k

    # :param input: Tensor of any dimension.
    # :param dim: (int or tuple of ints) The dimension or dimensions to reduce.
    # :param keepdim: (bool) Whether the output tensor has dim retained or not.
    # :param k: (float>0) How similar softmin is to min (the lower the more smooth).
    # """
    # return -torch.log(torch.sum(torch.exp(-k*input), dim, keepdim))/k
    """
    The generalized mean. It corresponds to the minimum when p = -inf.
    https://en.wikipedia.org/wiki/Generalized_mean
    :param tensor: Tensor of any dimension.
    :param dim: (int or tuple of ints) The dimension or dimensions to reduce.
    :param keepdim: (bool) Whether the output tensor has dim retained or not.
    :param p: (float<0).
    """
    assert p < 0
    res= torch.mean((tensor + 1e-6)**p, dim, keepdim=keepdim)**(1./p)
    return res

class linStack(torch.nn.Module):
    """A stack of linear layers followed by batch norm and hardTanh

    Attributes:
        num_layers: the number of linear layers.
        in_dim: the size of the input sample.
        hidden_dim: the size of the hidden layers.
        out_dim: the size of the output.
    """
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, bias, actBool, dp):
        super().__init__()

        layers_lin = []
        for i in range(num_layers):
            m = torch.nn.Linear(hidden_dim if i > 0 else in_dim,
                hidden_dim if i < num_layers - 1 else out_dim, bias=bias)
            layers_lin.append(m)
        self.layersLin = torch.nn.ModuleList(layers_lin)
        self.act_func = torch.nn.SELU()
        self.actBool = actBool
        self.dp = torch.nn.Dropout(p=dp)

    def forward(self, x):
        # Input shape (batch, features, *)
        for i, _ in enumerate(self.layersLin):
            x = self.act_func(x) if self.actBool else x
            x = self.layersLin[i](x)
            x = self.dp(x)
        return x

class regressionModule(torch.nn.Module):
    def __init__(self, feature_channels):
        super(regressionModule, self).__init__()
        if (type(feature_channels) == type(2)):
            inChannels = feature_channels
        else:
            inChannels = feature_channels['enc']['op'][-1]
        self.max_pool = nn.AvgPool2d(kernel_size=2)

        self.c1 = nn.Conv2d(in_channels=inChannels,
                            out_channels=128,
                            bias=True,
                            kernel_size=(2,3))

        self.c2 = nn.Conv2d(in_channels=128,
                            out_channels=128,
                            bias=True,
                            kernel_size=3)

        self.c3 = nn.Conv2d(in_channels=128,
                            out_channels=32,
                            kernel_size=3,
                            bias=False)

        self.l1 = nn.Linear(32*3*5, 256, bias=True)
        self.l2 = nn.Linear(256, 10, bias=True)

        self.c_actfunc = torch.tanh # Center has to be between -1 and 1
        self.param_actfunc = torch.sigmoid # Parameters can't be negative and capped to 1

    def forward(self, x, alpha):
        B = x.shape[0]
        # x: [B, 192, H/16, W/16]
        x = F.leaky_relu(self.c1(x)) # [B, 256, 14, 18]
        x = self.max_pool(x) # [B, 256, 7, 9]
        x = F.leaky_relu(self.c2(x)) # [B, 256, 5, 7]
        x = F.leaky_relu(self.c3(x)) # [B, 32, 3, 5]
        x = x.reshape(B, -1)
        x = self.l2(torch.selu(self.l1(x)))

        pup_c = self.c_actfunc(x[:, 0:2])
        pup_param = self.param_actfunc(x[:, 2:4])
        pup_angle = x[:, 4]
        iri_c = self.c_actfunc(x[:, 5:7])
        iri_param = self.param_actfunc(x[:, 7:9])
        iri_angle = x[:, 9]


        op = torch.cat([pup_c,
                        pup_param,
                        pup_angle.unsqueeze(1),
                        iri_c,
                        iri_param,
                        iri_angle.unsqueeze(1)], dim=1)
        return op

class convBlock(nn.Module):
    def __init__(self, in_c, inter_c, out_c, actfunc):
        super(convBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, inter_c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(inter_c, out_c, kernel_size=3, padding=1)
        self.actfunc = actfunc
        self.bn = torch.nn.BatchNorm2d(num_features=out_c)
    def forward(self, x):
        x = self.actfunc(self.conv1(x))
        x = self.actfunc(self.conv2(x)) # Remove x if not working properly
        x = self.bn(x)
        return x
class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

class Conv2dBlock(nn.Module):#Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
