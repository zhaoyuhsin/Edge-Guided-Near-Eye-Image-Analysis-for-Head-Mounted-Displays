#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rakshit
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import normPts, regressionModule, linStack, convBlock, LinearBlock, Conv2dBlock
from loss import conf_Loss, get_ptLoss, get_seg2ptLoss, get_segLoss, get_selfConsistency


def getSizes(chz, growth, blks=4):
    # This function does not calculate the size requirements for head and tail

    # Encoder sizes
    sizes = {'enc': {'inter': [], 'ip': [], 'op': []},
             'dec': {'skip': [], 'ip': [], 'op': []}}
    sizes['enc']['inter'] = np.array([chz * (i + 1) for i in range(0, blks)])
    sizes['enc']['op'] = np.array([np.int(growth * chz * (i + 1)) for i in range(0, blks)])
    sizes['enc']['ip'] = np.array([chz] + [np.int(growth * chz * (i + 1)) for i in range(0, blks - 1)])

    # Decoder sizes
    sizes['dec']['skip'] = sizes['enc']['ip'][::-1] + sizes['enc']['inter'][::-1]
    sizes['dec']['ip'] = sizes['enc']['op'][::-1]  # + sizes['dec']['skip']
    sizes['dec']['op'] = np.append(sizes['enc']['op'][::-1][1:], chz)
    return sizes


class Transition_down(nn.Module):
    def __init__(self, in_c, out_c, down_size, norm, actfunc):
        super(Transition_down, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.max_pool = nn.AvgPool2d(kernel_size=down_size) if down_size else False
        self.norm = norm(num_features=in_c)
        self.actfunc = actfunc

    def forward(self, x):
        x = self.actfunc(self.norm(x))
        x = self.conv(x)
        x = self.max_pool(x) if self.max_pool else x
        return x


class DenseNet2D_down_block(nn.Module):
    def __init__(self, in_c, inter_c, op_c, down_size, norm, actfunc):
        super(DenseNet2D_down_block, self).__init__()
        self.conv1 = nn.Conv2d(in_c, inter_c, kernel_size=3, padding=1)
        self.conv21 = nn.Conv2d(in_c + inter_c, inter_c, kernel_size=1, padding=0)
        self.conv22 = nn.Conv2d(inter_c, inter_c, kernel_size=3, padding=1)
        self.conv31 = nn.Conv2d(in_c + 2 * inter_c, inter_c, kernel_size=1, padding=0)
        self.conv32 = nn.Conv2d(inter_c, inter_c, kernel_size=3, padding=1)
        self.actfunc = actfunc
        self.bn = norm(num_features=in_c)
        self.TD = Transition_down(inter_c + in_c, op_c, down_size, norm, actfunc)

    def forward(self, x):
        x1 = self.actfunc(self.conv1(self.bn(x)))
        x21 = torch.cat([x, x1], dim=1)
        x22 = self.actfunc(self.conv22(self.conv21(x21)))
        x31 = torch.cat([x21, x22], dim=1)
        out = self.actfunc(self.conv32(self.conv31(x31)))
        out = torch.cat([out, x], dim=1)
        return out, self.TD(out)


class DenseNet2D_up_block(nn.Module):
    def __init__(self, skip_c, in_c, out_c, up_stride, actfunc):
        super(DenseNet2D_up_block, self).__init__()
        self.conv11 = nn.Conv2d(skip_c + in_c, out_c, kernel_size=1, padding=0)
        self.conv12 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.conv21 = nn.Conv2d(skip_c + in_c + out_c, out_c, kernel_size=1, padding=0)
        self.conv22 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.actfunc = actfunc
        self.up_stride = up_stride

    def forward(self, prev_feature_map, x):
        x = F.interpolate(x,
                          mode='bilinear',
                          align_corners=False,
                          scale_factor=self.up_stride)
        x = torch.cat([x, prev_feature_map], dim=1)
        x1 = self.actfunc(self.conv12(self.conv11(x)))
        x21 = torch.cat([x, x1], dim=1)
        out = self.actfunc(self.conv22(self.conv21(x21)))
        return out


class StyleEncoder(nn.Module):  # style encoder  4, input_dim = 3, dim = 64, style_dim = 8
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)]  # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):  # MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):
        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')]  # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class DenseNet_encoder(nn.Module):
    def __init__(self, in_c=1, chz=32, actfunc=F.leaky_relu, growth=1.5, norm=nn.BatchNorm2d):
        super(DenseNet_encoder, self).__init__()
        sizes = getSizes(chz, growth)
        interSize = sizes['enc']['inter']
        opSize = sizes['enc']['op']
        ipSize = sizes['enc']['ip']

        self.head = convBlock(in_c=in_c,
                              inter_c=chz,
                              out_c=chz,
                              actfunc=actfunc)
        self.down_block1 = DenseNet2D_down_block(in_c=ipSize[0],
                                                 inter_c=interSize[0],
                                                 op_c=opSize[0],
                                                 down_size=2,
                                                 norm=norm,
                                                 actfunc=actfunc)
        self.down_block2 = DenseNet2D_down_block(in_c=ipSize[1],
                                                 inter_c=interSize[1],
                                                 op_c=opSize[1],
                                                 down_size=2,
                                                 norm=norm,
                                                 actfunc=actfunc)
        self.down_block3 = DenseNet2D_down_block(in_c=ipSize[2],
                                                 inter_c=interSize[2],
                                                 op_c=opSize[2],
                                                 down_size=2,
                                                 norm=norm,
                                                 actfunc=actfunc)
        self.down_block4 = DenseNet2D_down_block(in_c=ipSize[3],
                                                 inter_c=interSize[3],
                                                 op_c=opSize[3],
                                                 down_size=2,
                                                 norm=norm,
                                                 actfunc=actfunc)
        self.bottleneck = DenseNet2D_down_block(in_c=opSize[3],
                                                inter_c=interSize[3],
                                                op_c=opSize[3],
                                                down_size=0,
                                                norm=norm,
                                                actfunc=actfunc)

    def forward(self, x):
        x = self.head(x)  # chz
        skip_1, x = self.down_block1(x)  # chz
        skip_2, x = self.down_block2(x)  # 2 chz
        skip_3, x = self.down_block3(x)  # 4 chz
        skip_4, x = self.down_block4(x)  # 8 chz
        _, x = self.bottleneck(x)
        return skip_4, skip_3, skip_2, skip_1, x


class DenseNet_decoder(nn.Module):
    def __init__(self, setting, chz, out_c, growth, actfunc=F.leaky_relu, norm=nn.BatchNorm2d):
        super(DenseNet_decoder, self).__init__()
        sizes = getSizes(chz, growth)
        skipSize = sizes['dec']['skip']
        opSize = sizes['dec']['op']
        ipSize = sizes['dec']['ip']
        if (setting['add_edge'] == 1):
            ipSize = [306, 180, 100, 62]
            opSize = [180, 100, 62, 32]
        self.up_block4 = DenseNet2D_up_block(skipSize[0], ipSize[0], opSize[0], 2, actfunc)
        self.up_block3 = DenseNet2D_up_block(skipSize[1], ipSize[1], opSize[1], 2, actfunc)
        self.up_block2 = DenseNet2D_up_block(skipSize[2], ipSize[2], opSize[2], 2, actfunc)
        self.up_block1 = DenseNet2D_up_block(skipSize[3], ipSize[3], opSize[3], 2, actfunc)

        self.final = convBlock(chz, chz, out_c, actfunc)

    def forward(self, skip4, skip3, skip2, skip1, x):
        x = self.up_block4(skip4, x)
        x = self.up_block3(skip3, x)
        x = self.up_block2(skip2, x)
        x = self.up_block1(skip1, x)
        o = self.final(x)
        return o


class DenseNet2D(nn.Module):
    def __init__(self,
                 setting,
                 chz=32,
                 growth=1.2,
                 actfunc=F.leaky_relu,
                 norm=nn.InstanceNorm2d,
                 selfCorr=False,
                 disentangle=False):
        super(DenseNet2D, self).__init__()

        self.sizes = getSizes(chz, growth)

        self.toggle = True
        self.selfCorr = selfCorr
        self.disentangle = disentangle
        self.disentangle_alpha = 2
        self.setting = setting
        input_channels = 1
        if self.setting['input_concat'] == 1:
            input_channels = 2
        self.enc = DenseNet_encoder(in_c=input_channels, chz=chz, actfunc=actfunc, growth=growth, norm=norm)
        self.dec = DenseNet_decoder(self.setting, chz=chz, out_c=3, actfunc=actfunc, growth=growth, norm=norm)
        feature_channels = self.setting['feature_channels']
        if (setting['add_edge'] == 1):
            print('!!!!! ADD EDGE MODULE......')
            feature_channels *= 2
            assert feature_channels == 306
        if (self.setting['add_seg'] == 1):
            print('!!! ADD AdaIN Module....')
            style_dim = self.setting['style_dim']
            self.seg_encoder = StyleEncoder(4, 3, 64, style_dim, norm='none', activ='relu', pad_type='reflect')
            self.mlp = MLP(style_dim, feature_channels * 2, 256, 3, norm='none', activ='relu')
        self.elReg = regressionModule(feature_channels)

        self._initialize_weights()

    def setDatasetInfo(self, numSets=2):
        # Produces a 1 layered MLP which directly maps bottleneck to the DS ID
        self.numSets = numSets
        self.dsIdentify_lin = linStack(num_layers=2,
                                       in_dim=self.sizes['enc']['op'][-1],
                                       hidden_dim=64,
                                       out_dim=numSets,
                                       bias=True,
                                       actBool=False,
                                       dp=0.0)

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4), size
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def forward(self,
                x,  # Input batch of images [B, 1, H, W]
                x_edge, # Edge of images [B, 1, H, W]
                target,  # Target semantic output of 3 classes [B, H, W]
                pupil_center,  # Pupil center [B, 2]
                elNorm,  # Normalized ellipse parameters [B, 2, 5]
                spatWts,  # Spatial weights for segmentation loss (boundary loss) [B, H, W]
                distMap,  # Distance map for segmentation loss (surface loss) [B, 3, H, W]
                cond,  # A condition array for each entry which marks its status [B, 4]
                ID,  # A Tensor containing information about the dataset or subset a entry
                alpha):  # Alpha score for various loss curicullum

        assert (self.setting['input_concat'] + self.setting['add_edge'] < 2), 'edge can use only 1 time!'
        B, _, H, W = x.shape

        if(self.setting['only_edge'] == 1):
            #print('test2')
            x = x_edge
        if(self.setting['input_concat'] == 1):
            x = torch.cat((x, x_edge), 1)
        x4, x3, x2, x1, x = self.enc(x)
        latent = torch.mean(x.flatten(start_dim=2), -1)  # [B, features]
        if(self.setting['add_edge'] == 1):
            if self.setting['add_edge'] == 1:
                x4_, x3_, x2_, x1_, x_add = self.enc(x_edge)
                x = torch.cat((x, x_add), 1)

        op = self.dec(x4, x3, x2, x1, x)
        if self.setting['add_seg'] == 1:
            softmx = nn.Softmax(dim=1)
            if (self.setting['seg_detach']):
                op_enc = self.seg_encoder(softmx(op.detach()))  # [B, style_dim]
            else:
                op_enc = self.seg_encoder(softmx(op))  # [B, style_dim]

            # print('op_enc.shape', op_enc.shape)
            adain_params = self.mlp(op_enc).view(B, 2, -1)  # [B, 2, 153]
            x_mean, x_std = self.calc_mean_std(x)  # x_mean [B, 153(306)] x [B, 153(306), W / 16, H / 16]
            size = x.size()
            # print('x_mean.size', x_mean.shape)
            # print('adain_params.shape : ', adain_params.shape)
            # print(x)
            normalized_x = (x - x_mean.expand(
                size)) / x_std.expand(size)
            # print('t1', adain_params[:, 0].shape)
            # print('expand : ', adain_params[:, 0].expand(size).shape)
            x = normalized_x * adain_params[:, 0].view(B, -1, 1, 1).expand(size) \
                + adain_params[:, 1].view(B, -1, 1, 1).expand(size)
        elOut = self.elReg(x, alpha)  # Linear regression to ellipse parameters

        # %%
        op_tup = get_allLoss(op,  # Output segmentation map
                             elOut,  # Predicted Ellipse parameters
                             target,  # Segmentation targets
                             pupil_center,  # Pupil center
                             elNorm,  # Normalized ellipse equation
                             spatWts,  # Spatial weights
                             distMap,  # Distance maps
                             cond,  # Condition
                             ID,  # Image and dataset ID
                             alpha)

        loss, pred_c_seg = op_tup

        # Uses ellipse center from segmentation but other params from regression
        #print('ritnet_v2.py pred_c_seg : ', pred_c_seg, pred_c_seg.shape)
        #print(elOut)

        # if(len(pred_c_seg.shape) == 2):
        #     elPred = torch.cat([pred_c_seg[:, 0], elOut[:, 2:5],
        #                         pred_c_seg[:, 1], elOut[:, 7:10]], dim=1)  # Bx5
        # else:

        elPred = torch.cat([pred_c_seg[:, 0, :], elOut[:, 2:5],
                             pred_c_seg[:, 1, :], elOut[:, 7:10]], dim=1)  # Bx5
        #print(elPred)

        # %%
        if self.selfCorr:
            loss_selfCorr = get_selfConsistency(op, elPred, 1 - cond[:, 1])
            loss += 10 * loss_selfCorr
            print(loss_selfCorr.item())
        if self.disentangle:
            pred_ds = self.dsIdentify_lin(latent)
            # Disentanglement procedure
            if self.toggle:
                # Primary loss + alpha*confusion
                loss += self.disentangle_alpha * conf_Loss(pred_ds,
                                                           ID.to(torch.long),
                                                           self.toggle)
            else:
                # Secondary loss
                loss = conf_Loss(pred_ds, ID.to(torch.long), self.toggle)
        return op, elPred, latent, loss.unsqueeze(0), elOut

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def get_allLoss(op,  # Network output
                elOut,  # Network ellipse regression output
                target,  # Segmentation targets
                pupil_center,  # Pupil center
                elNorm,  # Normalized ellipse parameters
                spatWts,
                distMap,
                cond,  # Condition matrix, 0 represents modality exists
                ID,
                alpha):
    B, C, H, W = op.shape
    loc_onlyMask = (1 - cond[:, 1]).to(torch.float32)  # GT mask present (True means mask exist)
    loc_onlyMask.requires_grad = False  # Ensure no accidental backprop

    # Segmentation to pupil center loss using center of mass
    l_seg2pt_pup, pred_c_seg_pup = get_seg2ptLoss(op[:, 2, ...],
                                                  normPts(pupil_center,
                                                          target.shape[1:]), temperature=4)

    # Segmentation to iris center loss using center of mass
    if torch.sum(loc_onlyMask):
        # Iris center is only present when GT masks are present. Note that
        # elNorm will hold garbage values. Those samples should not be backprop
        iriMap = -op[:, 0, ...]  # Inverse of background mask
        l_seg2pt_iri, pred_c_seg_iri = get_seg2ptLoss(iriMap,
                                                      elNorm[:, 0, :2],
                                                      temperature=4)
        temp = torch.stack([loc_onlyMask, loc_onlyMask], dim=1)
        l_seg2pt_iri = torch.sum(l_seg2pt_iri * temp) / torch.sum(temp.to(torch.float32))
        l_seg2pt_pup = torch.mean(l_seg2pt_pup)

    else:
        # If GT map is absent, loss is set to 0.0
        # Set Iris and Pupil center to be same
        l_seg2pt_iri = 0.0
        l_seg2pt_pup = torch.mean(l_seg2pt_pup)
        pred_c_seg_iri = torch.clone(elOut[:, 5:7])
    if(len(pred_c_seg_pup.shape) == 1):
        # print('assert ...')
        pred_c_seg_pup = pred_c_seg_pup.unsqueeze(0)
        pred_c_seg_iri = pred_c_seg_iri.unsqueeze(0)
    pred_c_seg = torch.stack([pred_c_seg_iri,
                              pred_c_seg_pup], dim=1)  # Iris first policy
    # print(pred_c_seg.shape, pred_c_seg)
    # print('RITNet_v2() pred_c_seg : ', pred_c_seg)
    l_seg2pt = 0.5 * l_seg2pt_pup + 0.5 * l_seg2pt_iri

    # Segmentation loss -> backbone loss
    l_seg = get_segLoss(op, target, spatWts, distMap, loc_onlyMask, alpha)

    # Bottleneck ellipse losses
    # NOTE: This loss is only activated when normalized ellipses do not exist
    l_pt = get_ptLoss(elOut[:, 5:7], normPts(pupil_center,
                                             target.shape[1:]), 1 - loc_onlyMask)

    # Compute ellipse losses - F1 loss for valid samples
    l_ellipse = get_ptLoss(elOut, elNorm.view(-1, 10), loc_onlyMask)

    total_loss = l_seg2pt + 20 * l_seg + 10 * (l_pt + l_ellipse)

    return (total_loss, pred_c_seg)
