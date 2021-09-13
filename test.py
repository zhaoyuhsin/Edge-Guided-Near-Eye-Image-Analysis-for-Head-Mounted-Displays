#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, cv2
import sys
import tqdm
import torch
import pickle
import resource
import numpy as np
import matplotlib.pyplot as plt
from calc_box_iou import calc_ell_bbox_iou
from skimage import draw
from args import parse_args
from pytorchtools import load_from_file
from torch.utils.data import DataLoader

from helperfunctions import mypause, stackall_Dict

from loss import get_seg2ptLoss

from utils import get_nparams, get_predictions, calc_edge, search_proper_parameter_iou
from utils import getSeg_metrics, getPoint_metric, generateImageGrid, unnormPts
from helperfunctions import my_ellipse
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048*10, rlimit[1]))


def calc_acc(args, testloader, model, edge_model, device, return_all = False, disp=0):


    ious = []
    ious_bySample = []

    dists_pupil_latent = []
    dists_pupil_seg = []

    dists_iris_latent = []
    dists_iris_seg = []

    bbiou_pupil = []
    bbiou_iris = []

    para_iris = []
    para_pupil = []

    model.eval()

    opDict = {'id': [], 'archNum': [], 'archName': [], 'code': [],
              'scores': {'iou': [], 'lat_dst': [], 'seg_dst': []},
              'pred': {'pup_latent_c': [],
                       'pup_seg_c': [],
                       'iri_latent_c': [],
                       'iri_seg_c': [],
                       'mask': []},
              'gt': {'pup_c': [], 'mask': []}}

    has_pupil = True
    has_iris = True
    has_latent = True
    search_flag = False
    bbox_iou_flag = False
    record_new_parameters = False
    if (args.model == 'deepvog'):
        has_iris = False
        has_latent = False
    # load iris index
    if(args.record_img == 1):
        filename = 'img/index_file/' + args.curObj + '_' + args.visual_dir + '_index.pkl'
        f = open(filename, 'rb')
        index = pickle.load(f)
        print('from ', filename, 'load ', len(index), ' items.')
        now = 0
        index = np.sort(index)
        print(index)
        f.close()
        if(args.curObj == 'LPW'):
            index = [130, 784, 981, 1098, 1156, 2785, 2968, 3490, 3700, 3781,
                     4317, 4740, 4932, 5446, 5894, 7007, 7733, 8968, 10159, 10816]
            ds_id = '3'
        elif(args.curObj == 'NVG'):
            index = [251, 1050, 1633, 1757, 2303, 3080, 3152, 3341, 3945, 3977,
                     4163, 4201, 4780, 4788, 5266, 7810, 8860, 9733, 9406, 10576] #9297
            ds_id = '1'
        elif (args.curObj == 'Ope'):
            index = [826, 1475, 1864, 1925, 2022, 2880, 3792, 4013, 4634, 4714,
                     5694, 6551, 6919, 7187, 7631, 7653, 7929, 8426, 9405, 10915]
            ds_id = '2'
        elif (args.curObj == 'Fuh'):
            index = [444, 578, 643, 845, 2310, 2455, 2551, 3191, 3287, 3431,
                     3875, 4000, 7865, 8157, 9309, 9350, 10011, 10017, 10025, 11141]
            ds_id = '4'
        else: assert 1 == 2, 'illegal args.curObj!'
        if(args.method == 'ritnet'):
            method_id = '3'
        elif(args.method == ''):
            method_id = '4'
        elif (args.method == 'edge'):
            method_id = '5'
        else: assert 1 == 2



    with torch.no_grad():
        print('calc_test')
        # '' -> baseline 'edge' -> ours  'ritnet' -> RITNet 'deepvog' -> DeepVOG
        for bt, batchdata in enumerate(tqdm.tqdm(testloader)):
            if(args.test_normal and bt > 10):break
            #print('!!!!!', iris_index[now])
            #print(args.batchsize)
            if(now >= len(index)):break
            if(args.record_img == 1 and index[now] >= (bt + 1) * args.batchsize):continue
            #print('now : ', bt * args.batchsize, ' find : ', iris_index[now])
            img, labels, spatialWeights, distMap, pupil_center, iris_center, elNorm, cond, imInfo, img_ori = batchdata
            img_edge = calc_edge(args, img, edge_model, device)
            # rt = torch.ones(img_edge.size()).to(device).to(args.prec)
            # img_edge = torch.where(img_edge >= 0.1, rt, img_edge)
            op_tup = model(img.to(device).to(args.prec),
                           img_edge.to(device).to(args.prec),
                           labels.to(device).long(),
                           pupil_center.to(device).to(args.prec),
                           elNorm.to(device).to(args.prec),
                           spatialWeights.to(device).to(args.prec),
                           distMap.to(device).to(args.prec),
                           cond.to(device).to(args.prec),
                           imInfo[:, 2].to(device).to(torch.long),  # Send DS #
                           0)

            output, elPred, _, _, elOut = op_tup
            if(args.model == 'deepvog'):
                elOut = elPred

            latent_iris_center = elOut[:, 0:2].detach().cpu().numpy()
            latent_pupil_center = elOut[:, 5:7].detach().cpu().numpy()

            seg_iris_center = elPred[:, 0:2].detach().cpu().numpy()
            seg_pupil_center = elPred[:, 5:7].detach().cpu().numpy()

            predict = get_predictions(output)
            _, ii = getSeg_metrics(labels.numpy(),
                                               predict.numpy(),
                                               cond[:, 1].numpy())[1:]
            # check if in the iris_list


            # research ellipse angle information
            B, _, H, W = img.shape
            H = np.array([[W / 2, 0, W / 2], [0, H / 2, H / 2], [0, 0, 1]])
            iris_bbiou = 0
            pupil_bbiou = 0
            new_para = []
            new_i_para = []

            if (args.record_img == 1):
                for i in range(args.batchsize):
                    id = bt * args.batchsize + i
                    if (id != index[now]): continue
                    print('!!!!!find it!', id)
                    # origin image
                    im = img_ori[i].cpu().numpy().squeeze()
                    im = np.stack([im for i in range(0, 3)], axis=2)
                    if(args.method == ''):
                        #cv2.imwrite(os.path.join('img', args.curObj, args.visual_dir, str(id) + '_img_' + args.method + '.jpg'), im)
                        cv2.imwrite(
                            os.path.join('img', str(now), ds_id + '_' + str(id) + '_1_img_' + args.method + '.png'),im)
                    if(args.method == 'deepvog'):
                        rr, cc = np.where(predict[i, ...] == 1)
                        im[rr, cc, ...] = np.array([36, 231, 253])
                    else:
                        rr, cc = np.where(predict[i, ...] == 1)
                        im[rr, cc, ...] = np.array([120, 183, 53])  # green 0 255 0
                        rr, cc = np.where(predict[i, ...] == 2)
                        im[rr, cc, ...] = np.array([36, 231, 253])  # red 0  0 255
                    #cv2.imwrite(os.path.join('img', args.curObj, args.visual_dir, str(id) + '_seg_' + args.method + '.jpg'), im)
                    if(args.method == 'ritnet'):
                        cv2.imwrite(os.path.join('img', str(now), ds_id + '_' + str(id) + '_' + method_id + '_seg.png'), im)
                    # rr, cc = np.where(predict[i, ...] == 0)
                    # im[rr, cc, ...] = np.array([0, 0, 0])  # black
                    #
                    # cv2.imwrite(os.path.join('img', args.curObj, args.visual_dir, str(id) + '_seg2_' + args.method + '.jpg'), im)

                    # draw edge
                    if(args.method == 'edge'):
                        I = img_edge[i].cpu().numpy().squeeze()
                        I *= 255
                        I = 255 - I
                        print(I.shape)
                        # cv2.imwrite(os.path.join('img', args.curObj, args.visual_dir,
                        #                          str(id) + '_edge_' + args.method + '.jpg'), I)
                        cv2.imwrite(os.path.join('img', str(now), ds_id + '_' + str(id) + '_2_edge.png'), I)

                    # draw ellipse elnorm
                    if(args.method == '' or args.method == 'edge'):

                        im = img_ori[i].cpu().numpy().squeeze()
                        im = np.stack([im for i in range(0, 3)], axis=2)
                        rr, cc = np.where(predict[i, ...] == 1)
                        im[rr, cc, ...] = np.array([120, 183, 53])  # green 0 255 0
                        rr, cc = np.where(predict[i, ...] == 2)
                        im[rr, cc, ...] = np.array([36, 231, 253])  # red 0  0 255
                        l_iris = my_ellipse(elPred[i, :5, ...].cpu().numpy()).transform(H)[0]
                        l_pupil = my_ellipse(elPred[i, 5:, ...].cpu().numpy()).transform(H)[0]
                        l_iris_gt = my_ellipse(elNorm[i, 0, ...].cpu().numpy()).transform(H)[0]
                        l_pupil_gt = my_ellipse(elNorm[i, 1, ...].cpu().numpy()).transform(H)[0]

                        cv2.ellipse(im, (int(l_iris[0]), int(l_iris[1])),
                                    (int(l_iris[2]), int(l_iris[3])),
                                    l_iris[4] / 3.14159 * 180, 0, 360, (255, 0, 0), 1, cv2.LINE_AA)  # 画椭圆
                        cv2.ellipse(im, (int(l_pupil[0]), int(l_pupil[1])),
                                    (int(l_pupil[2]), int(l_pupil[3])),
                                    l_pupil[4] / 3.14159 * 180, 0, 360, (0, 0, 255), 1, cv2.LINE_AA)  # 画椭圆
                        # cv2.imwrite(os.path.join('img', args.curObj, args.visual_dir,
                        #                          str(id) + '_elnorm_' + args.method + '.jpg'), im)
                        if(method_id == '4'):cv2.imwrite(os.path.join('img', str(now), ds_id + '_' + str(id) + '_' + method_id + '_elnorm.png'), im)
                        im = img_ori[i].cpu().numpy().squeeze()
                        im = np.stack([im for i in range(0, 3)], axis=2)
                        rr, cc = np.where(labels[i, ...] == 1)
                        im[rr, cc, ...] = np.array([120, 183, 53])  # green 0 255 0
                        rr, cc = np.where(labels[i, ...] == 2)
                        im[rr, cc, ...] = np.array([36, 231, 253])  # red 0  0 255
                        cv2.ellipse(im, (int(l_iris_gt[0]), int(l_iris_gt[1])),
                                    (int(l_iris_gt[2]), int(l_iris_gt[3])),
                                    l_iris_gt[4] / 3.14159 * 180, 0, 360, (255, 0, 0), 1, cv2.LINE_AA)  # 画椭圆
                        cv2.ellipse(im, (int(l_pupil_gt[0]), int(l_pupil_gt[1])),
                                    (int(l_pupil_gt[2]), int(l_pupil_gt[3])),
                                    l_pupil_gt[4] / 3.14159 * 180, 0, 360, (0, 0, 255), 1, cv2.LINE_AA)  # 画椭圆

                        if (method_id == '5'): cv2.imwrite(
                            os.path.join('img', str(now), ds_id + '_' + str(id) + '_6_gt.png'), im)

                        # cv2.imwrite(
                        #     os.path.join('img', args.curObj, args.visual_dir,
                        #                  str(id) + '_seg_gt_' + args.method + '.jpg'),
                        #     im)
                        print('test')


                    # After search & draw searched elnorm
                    if (args.method == 'edge'):
                        im = img_ori[i].cpu().numpy().squeeze()
                        im = np.stack([im for i in range(0, 3)], axis=2)
                        rr, cc = np.where(predict[i, ...] == 1)
                        im[rr, cc, ...] = np.array([120, 183, 53])  # green 0 255 0
                        rr, cc = np.where(predict[i, ...] == 2)
                        im[rr, cc, ...] = np.array([36, 231, 253])  # red 0  0 255
                        _, new_iris_para = search_proper_parameter_iou((predict[i] == 1).cuda(),
                                                                       elPred[i, :5, ...].cpu().numpy(),
                                                                       elNorm[i, 0, ...].cpu().numpy(),
                                                                       l_iris,
                                                                       l_iris_gt)
                        _, new_pupil_para = search_proper_parameter_iou((predict[i] == 2).cuda(),
                                                                        elPred[i, 5:, ...].cpu().numpy(),
                                                                        elNorm[i, 1, ...].cpu().numpy(),
                                                                        l_pupil, l_pupil_gt)
                        # print(new_iris_para)
                        # [rr_i, cc_i] = draw.ellipse_perimeter(int(new_iris_para[1]),
                        #                                       int(new_iris_para[0]),
                        #                                       int(new_iris_para[3]),
                        #                                       int(new_iris_para[2]),
                        #                                       orientation=new_iris_para[4])
                        # [rr_p, cc_p] = draw.ellipse_perimeter(int(new_pupil_para[1]),
                        #                                       int(new_pupil_para[0]),
                        #                                       int(new_pupil_para[3]),
                        #                                       int(new_pupil_para[2]),
                        #                                       orientation=new_pupil_para[4])
                        # rr_i = rr_i.clip(6, im.shape[0] - 6)
                        # rr_p = rr_p.clip(6, im.shape[0] - 6)
                        # c_i = cc_i.clip(6, im.shape[1] - 6)
                        # cc_p = cc_p.clip(6, im.shape[1] - 6)
                        #
                        # im[rr_i, cc_i, ...] = np.array([255, 0, 0])
                        # im[rr_p, cc_p, ...] = np.array([0, 0, 255])
                        cv2.ellipse(im, (int(new_iris_para[0]), int(new_iris_para[1])),
                                    (int(new_iris_para[2]), int(new_iris_para[3])),
                                    new_iris_para[4] / 3.14159 * 180, 0, 360, (255, 0, 0), 1, cv2.LINE_AA)  # 画椭圆
                        cv2.ellipse(im, (int(new_pupil_para[0]), int(new_pupil_para[1])),
                                    (int(new_pupil_para[2]), int(new_pupil_para[3])),
                                    new_pupil_para[4] / 3.14159 * 180, 0, 360, (0, 0, 255), 1, cv2.LINE_AA)  # 画椭圆
                        # cv2.imwrite(os.path.join('img', args.curObj, args.visual_dir,
                        #                          str(id) + '_elnorm_search_' + args.method + '.jpg'), im)
                        if (method_id == '5'): cv2.imwrite(
                            os.path.join('img', str(now), ds_id + '_' + str(id) + '_' + method_id + '_elnorm.png'), im)

                        # # draw seg gt
                        # im = img_ori[i].cpu().numpy().squeeze()
                        # im = np.stack([im for i in range(0, 3)], axis=2)
                        # rr, cc = np.where(labels[i, ...] == 1)
                        # im[rr, cc, ...] = np.array([120, 183, 53])  # green 0 255 0
                        # rr, cc = np.where(labels[i, ...] == 2)
                        # im[rr, cc, ...] = np.array([36, 231, 253])  # red 0  0 255
                        # cv2.ellipse(im, (int(l_iris_gt[0]), int(l_iris_gt[1])),
                        #             (int(l_iris_gt[2]), int(l_iris_gt[3])),
                        #             l_iris_gt[4] / 3.14159 * 180, 0, 360, (255, 0, 0), 1, cv2.LINE_AA)  # 画椭圆
                        # cv2.ellipse(im, (int(l_pupil_gt[0]), int(l_pupil_gt[1])),
                        #             (int(l_pupil_gt[2]), int(l_pupil_gt[3])),
                        #             l_pupil_gt[4] / 3.14159 * 180, 0, 360, (0, 0, 255), 1, cv2.LINE_AA)  # 画椭圆
                        #
                        # cv2.imwrite(
                        #     os.path.join('img', args.curObj, args.visual_dir, str(id) + '_seg_gt_' + args.method + '.jpg'),
                        #     im)
                        #
                        #
                        # # draw elnorm gt
                        # im = img_ori[i].cpu().numpy().squeeze()
                        # im = np.stack([im for i in range(0, 3)], axis=2)
                        #
                        # l_iris = my_ellipse(elNorm[i, 0, ...].cpu().numpy()).transform(H)[0]
                        # l_pupil = my_ellipse(elNorm[i, 1, ...].cpu().numpy()).transform(H)[0]
                        # [rr_i, cc_i] = draw.ellipse_perimeter(int(l_iris[1]),
                        #                                       int(l_iris[0]),
                        #                                       int(l_iris[3]),
                        #                                       int(l_iris[2]),
                        #                                       orientation=l_iris[4])
                        # [rr_p, cc_p] = draw.ellipse_perimeter(int(l_pupil[1]),
                        #                                       int(l_pupil[0]),
                        #                                       int(l_pupil[3]),
                        #                                       int(l_pupil[2]),
                        #                                       orientation=l_pupil[4])
                        # rr_i = rr_i.clip(6, im.shape[0] - 6)
                        # rr_p = rr_p.clip(6, im.shape[0] - 6)
                        # cc_i = cc_i.clip(6, im.shape[1] - 6)
                        # cc_p = cc_p.clip(6, im.shape[1] - 6)
                        #
                        # im[rr_i, cc_i, ...] = np.array([255, 0, 0])
                        # im[rr_p, cc_p, ...] = np.array([0, 0, 255])
                        # cv2.imwrite(os.path.join('img', args.curObj, args.visual_dir,
                        #                          str(id) + '_elnorm_gt_' + args.method + '.jpg'), im)

                    now += 1
                    print(ii[i])
                    if (now >= len(index)): break

            if(bbox_iou_flag):
                for i in range(B):
                    # print(i)
                    l_iris = my_ellipse(elPred[i, :5, ...].cpu().numpy()).transform(H)[0]
                    l_iris_gt = my_ellipse(elNorm[i, 0, ...].cpu().numpy()).transform(H)[0]
                    l_pupil = my_ellipse(elPred[i, 5:, ...].cpu().numpy()).transform(H)[0]
                    l_pupil_gt = my_ellipse(elNorm[i, 1, ...].cpu().numpy()).transform(H)[0]
                    if(not search_flag):
                        iris_bbiou += calc_ell_bbox_iou(l_iris[:5], l_iris_gt[:5])
                        pupil_bbiou += calc_ell_bbox_iou(l_pupil[:5], l_pupil_gt[:5])
                        para_iris.append(abs(np.array(l_iris[:5]) - np.array(l_iris_gt[:5])))
                        para_pupil.append(abs(np.array(l_pupil[:5]) - np.array(l_pupil_gt[:5])))
                    else:
                        new_iris_iou, new_iris_para = search_proper_parameter_iou((predict[i] == 1).cuda(),
                                                                                  elPred[i, :5, ...].cpu().numpy(),
                                                                                  elNorm[i, 0, ...].cpu().numpy(),
                                                                                  l_iris,
                                                                                  l_iris_gt)
                        iris_bbiou += new_iris_iou
                        new_pupil_iou, new_pupil_para = search_proper_parameter_iou((predict[i] == 2).cuda(),
                                                                                    elPred[i, 5:, ...].cpu().numpy(),
                                                                                    elNorm[i, 1, ...].cpu().numpy(),
                                                                                    l_pupil, l_pupil_gt)
                        pupil_bbiou += new_pupil_iou
                        # para_iris.append(abs(np.array(new_iris_para) - np.array(l_iris_gt[:5])))
                        para_pupil.append(abs(np.array(new_pupil_para) - np.array(l_pupil_gt[:5])))

                        if(record_new_parameters):
                            new_para.append(new_pupil_para)
                            new_para.append(new_iris_para)
                if (record_new_parameters):
                    new_para = np.array(new_para)
                    new_para = new_para.reshape(-1, 5)

                iris_bbiou /= B
                pupil_bbiou /= B

                bbiou_iris.append(iris_bbiou)
                bbiou_pupil.append(pupil_bbiou)

            if(args.model == 'deepvog'):
                labels = (labels == 2).to(torch.long)
            iou, iou_bySample = getSeg_metrics(labels.numpy(),
                                               predict.numpy(),
                                               cond[:, 1].numpy())[1:]

            pup_seg_c = unnormPts(seg_pupil_center,
                                  img.shape[2:])
            dispI = generateImageGrid(img.cpu().numpy().squeeze(),
                                      predict.numpy(),
                                      elPred.detach().cpu().numpy().reshape(-1, 2, 5),
                                      pup_seg_c,
                                      cond.numpy(),
                                      elNorm.cpu().numpy().reshape(-1, 2, 5),
                                      #new_para,
                                      override=True,
                                      heatmaps=False)

            if args.disp:
                if bt == 0:
                    h_im = plt.imshow(dispI.permute(1, 2, 0))
                    plt.pause(5)
                else:
                    h_im.set_data(dispI.permute(1, 2, 0))
                    mypause(5)


            # print('calc pupil_center : ', pupil_center)
            # print('latent_pupil_center : ', latent_pupil_center)
            if (has_latent):
                latent_pupil_dist, latent_pupil_dist_bySample = getPoint_metric(pupil_center.numpy(),
                                                                                latent_pupil_center,
                                                                                cond[:, 0].numpy(),
                                                                                img.shape[2:],
                                                                                True)  # Unnormalizes the points
                latent_iris_dist, latent_iris_dist_bySample = getPoint_metric(iris_center.numpy(),
                                                                              latent_iris_center,
                                                                              cond[:, 1].numpy(),
                                                                              img.shape[2:],
                                                                              True)  # Unnormalizes the points
                dists_pupil_latent.append(latent_pupil_dist)
                dists_iris_latent.append(latent_iris_dist)

            seg_pupil_dist, seg_pupil_dist_bySample = getPoint_metric(pupil_center.numpy(),
                                                                      seg_pupil_center,
                                                                      cond[:, 1].numpy(),
                                                                      img.shape[2:],
                                                                      True)  # Unnormalizes the points

            dists_pupil_seg.append(seg_pupil_dist)

            if (has_iris):
                seg_iris_dist, seg_iris_dist_bySample = getPoint_metric(iris_center.numpy(),
                                                                        seg_iris_center,
                                                                        cond[:, 1].numpy(),
                                                                        img.shape[2:],
                                                                        True)  # Unnormalizes the points
                dists_iris_seg.append(seg_iris_dist)
            ious.append(iou)
            ious_bySample.append(iou_bySample)

        ious = np.stack(ious, axis=0)
        if(args.record_iou == 1):
            ious_bySample = np.stack(ious_bySample, axis=0)
            print(ious_bySample.shape)
            ious_bySample = np.resize(ious_bySample, (ious_bySample.shape[0] * ious_bySample.shape[1], 3))

            print(ious_bySample.shape)
            filename = os.path.join('img', args.curObj + '_' + args.method +'_ious.pkl')
            f = open(filename, 'wb')
            pickle.dump(ious_bySample, f)
            f.close()
            print('!!!ious result dump to ', filename, '....')
            print('shape : ', ious_bySample.shape)
        ious = np.nanmean(ious, axis=0)

        #print(np.array(bbiou_iris).shape, np.array(bbiou_pupil).shape, np.array(para_iris).shape, np.array(para_pupil).shape)
        print('mIoU: {}. IoUs: {}'.format(np.mean(ious), ious))
        if(has_latent): print('Latent space PUPIL dist. Mean: {}'.format(np.nanmean(dists_pupil_latent)))
        print('Segmentation PUPIL dist. Mean: {}'.format(np.nanmean(dists_pupil_seg)))
        if(has_latent): print('Latent space IRIS dist. Mean: {}'.format(np.nanmean(dists_iris_latent)))
        if(has_iris):print('Segmentation IRIS dist. Mean: {}'.format(np.nanmean(dists_iris_seg)))
        if (bbox_iou_flag):
            print('Bounding box Pupil IoU. Mean : {}'.format(np.nanmean(bbiou_pupil)))
            print('Bounding box Iris IoU. Mean : {}'.format(np.nanmean(bbiou_iris)))
            para_pupil = np.nanmean(para_pupil, 0)
            para_pupil[4] *= 180.0 / 3.1415
            para_iris = np.nanmean(para_iris, 0)
            para_iris[4] *= 180.0 / 3.1415
            print('ABS Pupil parameters : ', para_pupil)
            print('ABS Iris parameters : ', para_iris)
        if (args.model == 'deepvog'):
            return np.mean(ious), np.nanmean(dists_pupil_seg), 0
        if(return_all): return ious, np.nanmean(dists_pupil_latent), np.nanmean(dists_iris_latent), \
                               np.nanmean(dists_pupil_seg), np.nanmean(dists_iris_seg)
        return np.mean(ious), np.nanmean(dists_pupil_latent), np.nanmean(dists_iris_latent)


#%%
if __name__ == '__main__':
    from modelSummary import model_dict
    args = parse_args()

    device=torch.device("cuda")
    torch.cuda.manual_seed(12)
    if torch.cuda.device_count() > 1:
        print('Moving to a multiGPU setup.')
        args.useMultiGPU = True
    else:
        args.useMultiGPU = False
    torch.backends.cudnn.deterministic=False

    if args.model not in model_dict:
        print("Model not found.")
        print("valid models are: {}".format(list(model_dict.keys())))
        exit(1)

    LOGDIR = os.path.join(os.getcwd(), 'logs', args.model, args.expname)
    path2model = os.path.join(LOGDIR, 'weights')
    path2checkpoint = os.path.join(LOGDIR, 'checkpoints')
    path2writer = os.path.join(LOGDIR, 'TB.lock')
    path2op = os.path.join(os.getcwd(), 'op', str(args.curObj))

    os.makedirs(LOGDIR, exist_ok=True)
    os.makedirs(path2model, exist_ok=True)
    os.makedirs(path2checkpoint, exist_ok=True)
    os.makedirs(path2writer, exist_ok=True)
    os.makedirs(path2op, exist_ok=True)

    model = model_dict[args.model]

    netDict = load_from_file([args.loadfile,
                              os.path.join(path2checkpoint, 'checkpoint.pt')])
    startEp = netDict['epoch'] if 'epoch' in netDict.keys() else 0
    if 'state_dict' in netDict.keys():
        model.load_state_dict(netDict['state_dict'])

    print('Parameters: {}'.format(get_nparams(model)))
    model = model if not args.useMultiGPU else torch.nn.DataParallel(model)
    model = model.to(device).to(args.prec)

    f = open(os.path.join('curObjects',
                          'baseline',
                          'cond_'+str(args.curObj)+'.pkl'), 'rb')

    _, _, testObj = pickle.load(f)
    testObj.path2data = os.path.join(args.path2data, 'Datasets', 'All')
    testObj.augFlag = False

    testloader = DataLoader(testObj,
                            batch_size=args.batchsize,
                            shuffle=False,
                            num_workers=args.workers,
                            drop_last=False)

    if args.disp:
        fig, axs = plt.subplots(nrows=1, ncols=1)
    #%%
    accLoss = 0.0
    imCounter = 0

    ious = []

    dists_pupil_latent = []
    dists_pupil_seg = []

    dists_iris_latent = []
    dists_iris_seg = []

    model.eval()

    opDict = {'id':[], 'archNum': [], 'archName': [], 'code': [],
              'scores':{'iou':[], 'lat_dst':[], 'seg_dst':[]},
              'pred':{'pup_latent_c':[],
                      'pup_seg_c':[],
                      'iri_latent_c':[],
                      'iri_seg_c':[],
                      'mask':[]},
              'gt':{'pup_c':[], 'mask':[]}}

    with torch.no_grad():
        for bt, batchdata in enumerate(tqdm.tqdm(testloader)):
            img, labels, spatialWeights, distMap, pupil_center, iris_center, elNorm, cond, imInfo = batchdata
            out_tup = model(img.to(device).to(args.prec),
                            labels.to(device).long(),
                            pupil_center.to(device).to(args.prec),
                            elNorm.to(device).to(args.prec),
                            spatialWeights.to(device).to(args.prec),
                            distMap.to(device).to(args.prec),
                            cond.to(device).to(args.prec),
                            imInfo[:, 2].to(device).to(torch.long),
                            0.5)

            output, elOut, latent, loss = out_tup

            latent_pupil_center = elOut[:, 0:2].detach().cpu().numpy()
            latent_iris_center  = elOut[:, 5:7].detach().cpu().numpy()

            _, seg_pupil_center = get_seg2ptLoss(output[:, 2, ...].cpu(), pupil_center, temperature=4)
            _, seg_iris_center  = get_seg2ptLoss(-output[:, 0, ...].cpu(), iris_center, temperature=4)

            loss = loss if args.useMultiGPU else loss.mean()

            accLoss += loss.detach().cpu().item()
            predict = get_predictions(output)

            iou, iou_bySample = getSeg_metrics(labels.numpy(),
                                               predict.numpy(),
                                               cond[:, 1].numpy())[1:]

            latent_pupil_dist, latent_pupil_dist_bySample = getPoint_metric(pupil_center.numpy(),
                                                                            latent_pupil_center,
                                                                            cond[:,0].numpy(),
                                                                            img.shape[2:],
                                                                            True) # Unnormalizes the points

            seg_pupil_dist, seg_pupil_dist_bySample = getPoint_metric(pupil_center.numpy(),
                                                                      seg_pupil_center,
                                                                      cond[:,1].numpy(),
                                                                      img.shape[2:],
                                                                      True) # Unnormalizes the points

            latent_iris_dist, latent_iris_dist_bySample = getPoint_metric(iris_center.numpy(),
                                                                          latent_iris_center,
                                                                          cond[:,1].numpy(),
                                                                          img.shape[2:],
                                                                          True) # Unnormalizes the points

            seg_iris_dist, seg_iris_dist_bySample = getPoint_metric(iris_center.numpy(),
                                                                    seg_iris_center,
                                                                    cond[:,1].numpy(),
                                                                    img.shape[2:],
                                                                    True) # Unnormalizes the points

            dists_pupil_latent.append(latent_pupil_dist)
            dists_iris_latent.append(latent_iris_dist)
            dists_pupil_seg.append(seg_pupil_dist)
            dists_iris_seg.append(seg_iris_dist)

            ious.append(iou)

            pup_latent_c = unnormPts(latent_pupil_center,
                                     img.shape[2:])
            pup_seg_c = unnormPts(seg_pupil_center,
                                  img.shape[2:])
            iri_latent_c = unnormPts(latent_iris_center,
                                     img.shape[2:])
            iri_seg_c = unnormPts(seg_iris_center,
                                  img.shape[2:])

            dispI = generateImageGrid(img.numpy().squeeze(),
                                      predict.numpy(),
                                      elOut.detach().cpu().numpy().reshape(-1, 2, 5),
                                      pup_seg_c,
                                      cond.numpy(),
                                      override=True,
                                      heatmaps=False)

            for i in range(0, img.shape[0]):
                archNum = testObj.imList[imCounter, 1]
                opDict['id'].append(testObj.imList[imCounter, 0])
                opDict['code'].append(latent[i,...].detach().cpu().numpy())

                opDict['archNum'].append(archNum)
                opDict['archName'].append(testObj.arch[archNum])

                opDict['pred']['pup_latent_c'].append(pup_latent_c[i, :])
                opDict['pred']['pup_seg_c'].append(pup_seg_c[i, :])
                opDict['pred']['iri_latent_c'].append(iri_latent_c[i, :])
                opDict['pred']['iri_seg_c'].append(iri_seg_c[i, :])

                if args.test_save_op_masks:
                    opDict['pred']['mask'].append(predict[i,...].numpy().astype(np.uint8))

                opDict['scores']['iou'].append(iou_bySample[i, ...])
                opDict['scores']['lat_dst'].append(latent_pupil_dist_bySample[i, ...])
                opDict['scores']['seg_dst'].append(seg_pupil_dist_bySample[i, ...])

                opDict['gt']['pup_c'].append(pupil_center[i,...].numpy())

                if args.test_save_op_masks:
                    opDict['gt']['mask'].append(labels[i,...].numpy().astype(np.uint8))

                imCounter+=1

            if args.disp:
                if bt == 0:
                    h_im = plt.imshow(dispI.permute(1, 2, 0))
                    plt.pause(0.01)
                else:
                    h_im.set_data(dispI.permute(1, 2, 0))
                    mypause(0.01)

        opDict = stackall_Dict(opDict)
        ious = np.stack(ious, axis=0)
        ious = np.nanmean(ious, axis=0)
        print('mIoU: {}. IoUs: {}'.format(np.mean(ious), ious))
        print('Latent space PUPIL dist. Med: {}, STD: {}'.format(np.nanmedian(dists_pupil_latent),
                                                            np.nanstd(dists_pupil_latent)))
        print('Segmentation PUPIL dist. Med: {}, STD: {}'.format(np.nanmedian(dists_pupil_seg),
                                                            np.nanstd(dists_pupil_seg)))
        print('Latent space IRIS dist. Med: {}, STD: {}'.format(np.nanmedian(dists_iris_latent),
                                                           np.nanstd(dists_iris_latent)))
        print('Segmentation IRIS dist. Med: {}, STD: {}'.format(np.nanmedian(dists_iris_seg),
                                                           np.nanstd(dists_iris_seg)))

        print('--- Saving output directory ---')
        f = open(os.path.join(path2op, 'opDict.pkl'), 'wb')
        pickle.dump(opDict, f)
        f.close()
