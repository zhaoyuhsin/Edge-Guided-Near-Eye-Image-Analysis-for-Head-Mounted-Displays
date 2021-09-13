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
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from helperfunctions import mypause, linVal
from pytorchtools import EarlyStopping, load_from_file
from utils import get_nparams, Logger, get_predictions, lossandaccuracy
from utils import getSeg_metrics, getPoint_metric, generateImageGrid, unnormPts
from utils import getAng_metric, calc_edge
from test import calc_acc
from bdcn_new import BDCN
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
    setting = get_config(args.setting)
    print('Setting : ')
    print(setting)
    device=torch.device("cuda")
    torch.cuda.manual_seed_all(12)

    if torch.cuda.device_count() > 1:
        print('Moving to a multiGPU setup.')
        args.useMultiGPU = True
    else:
        print('Single GPU setup')
        args.useMultiGPU = False


    LOGDIR          = os.path.join(os.getcwd(), 'logs', args.model, args.expname)
    path2model      = os.path.join(LOGDIR, 'weights')
    path2writer     = os.path.join(LOGDIR, 'TB.lock')
    path2checkpoint = os.path.join(LOGDIR, 'checkpoints')
    path2pretrained = os.path.join(os.getcwd(),
                                   'logs',
                                   args.model,
                                   'pretrained',
                                   'weights',
                                   'pretrained.git_ok')

    # Generate directories if they don't exist
    os.makedirs(LOGDIR, exist_ok=True)
    os.makedirs(path2model, exist_ok=True)
    os.makedirs(path2checkpoint, exist_ok=True)
    os.makedirs(path2writer, exist_ok=True)

    # Open relevant train/test object
    f = open(os.path.join(args.test_mode,'cond_'+str(args.curObj)+'.pkl'), 'rb')
    trainObj, validObj, _ = pickle.load(f)
    trainObj.augFlag = True
    validObj.augFlag = False

    ff = open(os.path.join('baseline', 'cond_LPW.pkl'), 'rb')
    lpw_validObj, _, lpw_testObj = pickle.load(ff)
    lpw_validObj.augFlag = False
    lpw_testObj.augFlag = False

    if(args.id == 0):
        trainObj.path2data = os.path.join(args.path2data, 'Datasets', 'All/New')
        validObj.path2data = os.path.join(args.path2data, 'Datasets', 'All/New')
        lpw_validObj.path2data = os.path.join(args.path2data, 'Datasets', 'TEyeD-h5-Edges')
        lpw_testObj.path2data = os.path.join(args.path2data, 'Datasets', 'TEyeD-h5-Edges')
    elif(args.id == 1):
        trainObj.path2data = os.path.join(args.path2data, 'userdata/riteye.zip/All/New')
        validObj.path2data = os.path.join(args.path2data, 'userdata/riteye.zip/All/New')
        lpw_validObj.path2data = os.path.join(args.path2data, 'userdata/lpw.zip/TEyeD-h5-Edges')
        lpw_testObj.path2data = os.path.join(args.path2data, 'userdata/lpw.zip/TEyeD-h5-Edges')
    else:
        assert(1 == 2), 'illeagel id'


    print(' train, valid, lpw_valid, lpw_testObj: ', len(trainObj), len(validObj), len(lpw_validObj), len(lpw_testObj))

    lpw_validloader = DataLoader(lpw_validObj,
                                 batch_size=args.batchsize,
                                 shuffle=False,
                                 drop_last=True,
                                 num_workers=args.workers)
    lpw_testloader = DataLoader(lpw_testObj,
                            batch_size=args.batchsize,
                            shuffle=False,
                            drop_last=True,
                            num_workers=args.workers)

    # load model
    BDCN_network = BDCN()

    state_dict = torch.load('gen_00000016.pt')
    BDCN_network.load_state_dict(state_dict['a'])
    BDCN_network = BDCN_network.cuda()
    BDCN_network.eval()

    writer = SummaryWriter(path2writer)
    logger = Logger(os.path.join(LOGDIR,'logs.log'))

    # Ensure model has all necessary weights initialized
    if args.model == 'ritnet_v1':
        model = RITNet()
    elif(args.model == 'ritnet_v2'):
        model = DenseNet2D(setting)
    elif (args.model == 'deepvog'):
        model = DeepVOG_pytorch()
    else: assert 1==2, 'illegal model'

    model.selfCorr = args.selfCorr
    model.disentangle = args.disentangle

    param_list = [param for name, param in model.named_parameters() if 'dsIdentify' not in name]
    optimizer = torch.optim.Adam([{'params':param_list,
                                   'lr':args.lr}]) # Set optimizer

    # If loading pretrained weights, ensure you don't load confusion branch
    if args.resume:
        print ("NOTE resuming training. Priority: 1) Checkpoint 2) Epoch #")
        checkpointfile = os.path.join(path2checkpoint, 'checkpoint.pt')

        model   = model.to(device)
        netDict = load_from_file([checkpointfile, args.loadfile])

        # Load previous checkpoint and resume from that epoch
        model.load_state_dict(netDict['state_dict'])
        startEp = netDict['epoch'] + 1 if 'epoch' in netDict.keys() else 0

    elif 'pretrained' not in args.expname:
        # If the very first epoch, then save out an _init pickle
        # This is particularly useful for lottery tickets
        print('Searching for pretrained weights ...')

        if os.path.exists(path2pretrained):
            netDict = torch.load(path2pretrained)
            model.load_state_dict(netDict['state_dict'])
            print('Pretrained weights loaded! Enjoy the ride ...')
        else:
            print('No pretrained. Warning. Training on only pupil centers leads to instability.')
        startEp = 0
        torch.save(model.state_dict(), os.path.join(path2model, args.model+'{}.pkl'.format('_init')))
    else:
        startEp = 0
        print('Pretraining mode detected ...')
        torch.save(model.state_dict(), os.path.join(path2model, args.model+'{}.pkl'.format('_init')))

    # Let the network know you need a disentanglement module.
    # Please refer to args.py for more information on disentanglement strategy
    if args.disentangle:
        # Let the model know how many datasets it must expect
        print('Total # of datasets found: {}'.format(np.unique(trainObj.imList[:, 2]).size))
        model.setDatasetInfo(np.unique(trainObj.imList[:, 2]).size)
        opt_disent = torch.optim.Adam(model.dsIdentify_lin.parameters(), lr=1*args.lr)

    nparams = get_nparams(model)
    print('Total number of trainable parameters: {}\n'.format(nparams))

    patience = 10
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           'max',
                                                           patience=patience-5,
                                                           verbose=True,
                                                           factor=0.1) # Default factor = 0.1

    early_stopping = EarlyStopping(mode='max',
                                   delta=0.001,
                                   verbose=True,
                                   patience=patience,
                                   fName='checkpoint.pt',
                                   path2save=path2checkpoint)

    model = model if not args.useMultiGPU else torch.nn.DataParallel(model)
    model = model.to(device).to(args.prec) # NOTE: good habit to do this before optimizer

    if args.overfit > 0:
        # This is a flag to check if attempting to overfit on a small subset
        # This is used as a quick check to verify training process
        trainObj.imList = trainObj.imList[:args.overfit*args.batchsize,:]
        validObj.imList = validObj.imList[:args.overfit*args.batchsize,:]

    trainloader = DataLoader(trainObj,
                             batch_size=args.batchsize,
                             shuffle=True,
                             num_workers=args.workers,
                             drop_last=True)
    validloader = DataLoader(validObj,
                             batch_size=args.batchsize,
                             shuffle=False,
                             num_workers=args.workers,
                             drop_last=True)

    if args.disp:
        fig, axs = plt.subplots(nrows=1, ncols=1)
    #%%
    print('!!!!!-----MODEL STRUCTURE------------')
    tot_parameters = 0
    for k, v in model.named_parameters():
        print(k, v.shape)
        tot_parameters += v.numel()
    print('tot_para: ', tot_parameters)
    print('------------------------------------\n\n\n')
    print('trainObj, validObj, lpw_trainObj, lpw_testObj', len(trainObj), len(validObj), len(lpw_validObj), len(lpw_testObj))
    print(args.workers)
    max_miou = 0.0
    min_pup_c = 30.0
    #miou_test, pct_test, ict_test = calc_acc(args, lpw_testloader, model, BDCN_network, device)

    # calc time for analyse
    start_time = time.time()
    calc_edge_time = 0
    calc_network_time = 0

    for epoch in range(startEp, args.epochs):
        accLoss = 0.0
        ious = []

        scoreType = {'c_dist':[], 'ang_dist': [], 'sc_rat': []}
        scoreTrack = {'pupil': copy.deepcopy(scoreType),
                      'iris': copy.deepcopy(scoreType)}

        model.train()
        alpha = linVal(epoch, (0, args.epochs), (0, 1), 0)
        lossandaccuracy(args,  # Training arguments
                        validloader,  # Validation loader
                        model,  # Model
                        BDCN_network,
                        alpha,  # Alpha value to measure loss
                        device)
        for bt, batchdata in enumerate(trainloader):
            if(args.test_normal and bt > 50):break
            img, labels, spatialWeights, distMap, pupil_center, iris_center, elNorm, cond, imInfo = batchdata
            start_edge = time.time()
            img_edge = calc_edge(args, img, BDCN_network, device)
            calc_edge_time += time.time() - start_edge

            optimizer.zero_grad()

            start_network = time.time()
            out_tup = model(img.to(device).to(args.prec),
                            img_edge,
                            labels.to(device).long(),
                            pupil_center.to(device).to(args.prec),
                            elNorm.to(device).to(args.prec),
                            spatialWeights.to(device).to(args.prec),
                            distMap.to(device).to(args.prec),
                            cond.to(device).to(args.prec),
                            imInfo[:, 2].to(device).to(torch.long), # Send DS #
                            alpha)
            calc_network_time += time.time() - start_network
            output, elOut, _, loss, _ = out_tup

            loss = loss.mean() if args.useMultiGPU else loss
            loss.backward()
            optimizer.step()

            # Predicted centers
            pred_c_iri = elOut[:, 0:2].detach().cpu().numpy()
            pred_c_pup = elOut[:, 5:7].detach().cpu().numpy()

            accLoss += loss.detach().cpu().item()
            predict = get_predictions(output)

            # IOU metric
            iou = getSeg_metrics(labels.numpy(),
                                 predict.numpy(),
                                 cond[:, 1].numpy())[1]
            ious.append(iou)

            # Center distance
            ptDist_iri = getPoint_metric(iris_center.numpy(),
                                         pred_c_iri,
                                         cond[:,1].numpy(),
                                         img.shape[2:],
                                         True)[0] # Unnormalizes the points
            ptDist_pup = getPoint_metric(pupil_center.numpy(),
                                         pred_c_pup,
                                         cond[:,0].numpy(),
                                         img.shape[2:],
                                         True)[0] # Unnormalizes the points

            # Angular distance
            angDist_iri = getAng_metric(elNorm[:, 0, 4].numpy(),
                                        elOut[:, 4].detach().cpu().numpy(),
                                        cond[:, 1].numpy())[0]
            angDist_pup = getAng_metric(elNorm[:, 1, 4].numpy(),
                                        elOut[:, 9].detach().cpu().numpy(),
                                        cond[:, 1].numpy())[0]

            # Scale metric
            gt_ab = elNorm[:, 0, 2:4]
            pred_ab = elOut[:, 2:4].cpu().detach()
            scale_iri = torch.sqrt(torch.sum(gt_ab**2, dim=1)/torch.sum(pred_ab**2, dim=1))
            scale_iri = torch.sum(scale_iri*(~cond[:,1]).to(torch.float32)).item()
            gt_ab = elNorm[:, 1, 2:4]
            pred_ab = elOut[:, 7:9].cpu().detach()
            scale_pup = torch.sqrt(torch.sum(gt_ab**2, dim=1)/torch.sum(pred_ab**2, dim=1))
            scale_pup = torch.sum(scale_pup*(~cond[:,1]).to(torch.float32)).item()

            # Append to score dictionary
            scoreTrack['iris']['c_dist'].append(ptDist_iri)
            scoreTrack['iris']['ang_dist'].append(angDist_iri)
            scoreTrack['iris']['sc_rat'].append(scale_iri)
            scoreTrack['pupil']['c_dist'].append(ptDist_pup)
            scoreTrack['pupil']['ang_dist'].append(angDist_pup)
            scoreTrack['pupil']['sc_rat'].append(scale_pup)

            iri_c = unnormPts(pred_c_iri,
                              img.shape[2:])
            pup_c = unnormPts(pred_c_pup,
                              img.shape[2:])

            if args.disp:
                # Generate image grid with overlayed predicted data
                dispI = generateImageGrid(img.squeeze().numpy(),
                          predict.numpy(),
                          elOut.detach().cpu().numpy().reshape(-1, 2, 5),
                          pup_c,
                          cond.numpy(),
                          elNorm.cpu().numpy(),
                          override=True,
                          heatmaps=False)
                if (epoch == startEp) and (bt == 0):
                    h_im = plt.imshow(dispI.permute(1, 2, 0))
                    plt.pause(0.01)
                else:
                    h_im.set_data(dispI.permute(1, 2, 0))
                    mypause(0.01)

            if bt%30 == 0:
                now_time = time.time()
                logger.write('Epoch:{} [{}/{}], Loss: {:.3f} time:{:.3f}'.format(epoch,
                                                                     bt,
                                                                     len(trainloader),
                                                                     loss.item(),
                                                                    now_time - start_time))
                print('calc_edge time : {:.3f}'.format(calc_edge_time))
                print('calc_network_time : {:.3f}'.format(calc_network_time))
                calc_edge_time = 0
                calc_network_time = 0
                start_time = now_time

        valid_st = time.time()

        # Sketch the very last batch. Training drops uneven batches..
        dispI = generateImageGrid(img.squeeze().numpy(),
                                  predict.numpy(),
                                  elOut.detach().cpu().numpy().reshape(-1, 2, 5),
                                  pup_c,
                                  cond.numpy(),
                                  elNorm.cpu().numpy(),
                                  override=True,
                                  heatmaps=False)

        ious = np.stack(ious, axis=0)
        ious = np.nanmean(ious, axis=0)
        logger.write('Epoch:{}, Train IoU: {}'.format(epoch, ious))

        out_tup = lossandaccuracy(args, # Training arguments
                                  validloader, # Validation loader
                                  model, # Model
                                  BDCN_network,
                                  alpha, # Alpha value to measure loss
                                  device)
        lossvalid, ious_valid, scoreTrack_v, latent_codes= out_tup
        print('valid_loader time : {:.3f}'.format(time.time() - valid_st))
        valid_st = time.time()

        # Add iris info to tensorboard
        writer.add_scalars('iri_c/mu', {'train':np.nanmean(scoreTrack['iris']['c_dist']),
                                         'valid':np.nanmean(scoreTrack_v['iris']['c_dist'])}, epoch)
        writer.add_scalars('iri_c/std', {'train':np.nanstd(scoreTrack['iris']['c_dist']),
                                          'valid':np.nanstd(scoreTrack_v['iris']['c_dist'])}, epoch)
        writer.add_scalars('iri_ang/mu', {'train':np.nanmean(scoreTrack['iris']['ang_dist']),
                                          'valid':np.nanmean(scoreTrack_v['iris']['ang_dist'])}, epoch)
        writer.add_scalars('iri_ang/std', {'train':np.nanstd(scoreTrack['iris']['ang_dist']),
                                          'valid':np.nanstd(scoreTrack_v['iris']['ang_dist'])}, epoch)
        writer.add_scalars('iri_sc/mu', {'train':np.nanmean(scoreTrack['iris']['sc_rat']),
                                          'valid':np.nanmean(scoreTrack_v['iris']['sc_rat'])}, epoch)
        writer.add_scalars('iri_sc/std', {'train':np.nanstd(scoreTrack['iris']['sc_rat']),
                                          'valid':np.nanstd(scoreTrack_v['iris']['sc_rat'])}, epoch)

        # Add pupil info to tensorboard
        writer.add_scalars('pup_c/mu', {'train':np.nanmean(scoreTrack['pupil']['c_dist']),
                                         'valid':np.nanmean(scoreTrack_v['pupil']['c_dist'])}, epoch)
        writer.add_scalars('pup_c/std', {'train':np.nanstd(scoreTrack['pupil']['c_dist']),
                                          'valid':np.nanstd(scoreTrack_v['pupil']['c_dist'])}, epoch)
        writer.add_scalars('pup_ang/mu', {'train':np.nanmean(scoreTrack['pupil']['ang_dist']),
                                          'valid':np.nanmean(scoreTrack_v['pupil']['ang_dist'])}, epoch)
        writer.add_scalars('pup_ang/std', {'train':np.nanstd(scoreTrack['pupil']['ang_dist']),
                                          'valid':np.nanstd(scoreTrack_v['pupil']['ang_dist'])}, epoch)
        writer.add_scalars('pup_sc/mu', {'train':np.nanmean(scoreTrack['pupil']['sc_rat']),
                                          'valid':np.nanmean(scoreTrack_v['pupil']['sc_rat'])}, epoch)
        writer.add_scalars('pup_sc/std', {'train':np.nanstd(scoreTrack['pupil']['sc_rat']),
                                          'valid':np.nanstd(scoreTrack_v['pupil']['sc_rat'])}, epoch)

        writer.add_scalar('loss/train', accLoss/bt, epoch)
        writer.add_scalar('loss/valid', lossvalid, epoch)

        # Write image to tensorboardX
        writer.add_image('train/op', dispI, epoch)

        if epoch%embed_log == 0:
            print('Saving validation embeddings ...')
            writer.add_embedding(torch.cat(latent_codes, 0),
                                 metadata=validObj.imList[:len(latent_codes)*args.batchsize, 2],
                                 global_step=epoch)

        f = 'Epoch:{}, Valid Loss: {:.3f}, mIoU: {}'
        logger.write(f.format(epoch, lossvalid, np.mean(ious)))

        # Generate a model dictionary which stores epochs and current state
        netDict = {'state_dict':[], 'epoch': epoch}
        stateDict = model.state_dict() if not args.useMultiGPU else model.module.state_dict()
        netDict['state_dict'] = {k: v for k, v in stateDict.items() if 'dsIdentify_lin' not in k}

        pup_c_dist = np.nanmean(scoreTrack_v['pupil']['c_dist'])
        pup_ang_dist = np.nanmean(scoreTrack_v['pupil']['ang_dist'])
        if not np.isnan(np.mean(ious)):
            iri_c_dist = np.nanmean(scoreTrack_v['iris']['c_dist'])
            iri_ang_dist = np.nanmean(scoreTrack_v['iris']['ang_dist'])
            stopMetric = np.mean(ious_valid) + 2 - 2.5e-3*(pup_c_dist + iri_c_dist) +\
                        (1 - pup_ang_dist/90) + (1 - iri_ang_dist/90) # Max value 5
        else:
            stopMetric = 1 - (pup_c_dist/400) # Max value 1
        scheduler.step(stopMetric)
        print('Now LR = ', optimizer.state_dict()['param_groups'][0]['lr'])
        early_stopping(stopMetric, netDict)

        if early_stopping.early_stop:
            torch.save(netDict, os.path.join(path2model, args.model + '_earlystop_{}.pkl'.format(epoch)))
            print("Early stopping")
            break

        print('start time : {:.3f}'.format(time.time() - valid_st))
        valid_st = time.time()
        print('!!!!Valid Process....')
        miou_valid2, pct_valid2, tg = calc_acc(args, lpw_validloader, model, BDCN_network, device)
        print('lpw_train loader time : {:.3f} minu'.format((time.time() - valid_st) / 60.0))
        valid_st = time.time()
        writer.add_scalar('LPW/mIou', miou_valid2 * 100.0, epoch)
        writer.add_scalar('LPW/Reg_pupil_center', pct_valid2, epoch)
        writer.add_scalar('LPW/Reg_iris_center', tg, epoch)
        if (miou_valid2 > max(max_miou, 0.7) or pct_valid2 <= min(min_pup_c, 25.0)):
            max_miou = max(max_miou, miou_valid2)
            min_pup_c = min(min_pup_c, pct_valid2)
            print('Test Process......Epoch : {}'.format(epoch))
            miou_test, pct_test, ict_test = calc_acc(args, lpw_testloader, model, BDCN_network, device)
            print('Successfully test. Epoch : {}'.format(epoch))
        print('lpw_test loader time : {:.3f} minu'.format((time.time() - valid_st) / 60.0))
        valid_st = time.time()

        ##save the model every 2 epochs
        if epoch%1 == 0:
            torch.save(netDict,
                       os.path.join(path2model, args.model+'_{}.pkl'.format(epoch)))
    writer.close()
