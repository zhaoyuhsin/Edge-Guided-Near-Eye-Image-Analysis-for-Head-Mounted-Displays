#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, cv2, sys, yaml, tqdm, torch, pickle, resource
import numpy as np
import matplotlib.pyplot as plt
from calc_box_iou import calc_ell_bbox_iou
from skimage import draw
from args import parse_args
from pytorchtools import load_from_file
from torch.utils.data import DataLoader
from bdcn_new import BDCN
from helperfunctions import mypause, stackall_Dict
from models.RITnet_v2 import DenseNet2D
from models.RITnet_v1 import DenseNet2D as RITNet
from models.deepvog_pytorch import DeepVOG_pytorch
from loss import get_seg2ptLoss

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)
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




    with torch.no_grad():
        # '' -> baseline 'edge' -> ours  'ritnet' -> RITNet 'deepvog' -> DeepVOG
        for bt, batchdata in enumerate(tqdm.tqdm(testloader)):
            if(args.test_normal and bt > 10):break

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
    args = parse_args()
    setting = get_config(args.setting)
    print('Setting : ')
    print(setting)
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(12)

    if torch.cuda.device_count() > 1:
        print('Moving to a multiGPU setup.')
        args.useMultiGPU = True
    else:
        print('Single GPU setup')
        args.useMultiGPU = False

    ff = open(os.path.join('baseline', 'cond_' + str(args.curObj) + '.pkl'), 'rb')
    _, _, lpw_testObj = pickle.load(ff)
    lpw_testObj.augFlag = False
    lpw_testObj.path2data = os.path.join(args.path2data, 'Datasets', 'TEyeD-h5-Edges')
    lpw_testloader = DataLoader(lpw_testObj,
                                batch_size=args.batchsize,
                                shuffle=False,
                                drop_last=True,
                                num_workers=args.workers)
    BDCN_network = BDCN()

    state_dict = torch.load('gen_00000016.pt')
    BDCN_network.load_state_dict(state_dict['a'])
    BDCN_network = BDCN_network.cuda()
    BDCN_network.eval()
    if args.model == 'ritnet_v1':
        model = RITNet()
    elif(args.model == 'ritnet_v2'):
        model = DenseNet2D(setting)
    elif (args.model == 'deepvog'):
        model = DeepVOG_pytorch()
    else: assert 1==2, 'illegal model'

    netDict = load_from_file([args.loadfile])
    model.load_state_dict(netDict['state_dict'])
    model = model if not args.useMultiGPU else torch.nn.DataParallel(model)
    model = model.to(device).to(args.prec)  # NOTE: good habit to do this before optimizer
    miou_test, pct_test, ict_test = calc_acc(args, lpw_testloader, model, BDCN_network, device)