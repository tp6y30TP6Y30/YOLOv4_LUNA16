#!/usr/bin/python3
#coding=utf-8

import torch
from torch import nn
import torch.nn.functional as F

from layers import GetPBB
from loss_2 import Loss_recon as Loss

config = {}
config['anchors'] = [5., 10., 20.] #[ 10.0, 30.0, 60.]
config['channel'] = 1
config['crop_size'] = [80, 80, 80]
config['stride'] = 4
config['max_stride'] = 16
config['num_neg'] = 800
config['th_neg'] = 0.02
config['th_pos_train'] = 0.5
config['th_pos_val'] = 1
config['num_hard'] = 2
config['bound_size'] = 12
config['reso'] = 1
config['sizelim'] = 2.5 #3 #6. #mm
config['sizelim2'] = 10 #30
config['sizelim3'] = 20 #40
config['sizelim4'] = 40.
config['aug_scale'] = True
config['r_rand_crop'] = 0.3
config['pad_value'] = 170
config['augtype'] = {'flip':True,'swap':False,'scale':True,'rotate':False}


config['augtype'] = {'flip':True,'swap':False,'scale':True,'rotate':False}
config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38', '990fbe3f0a1b53878669967b9afd1441', 'adc3bbc63d40f8761c59be10f1e504c3',
                       'f938f9022abf7f1072fe9df79db7eccd', 'eb008af181f3791fdce2376cf4773733', '11fe5426ef497bc490b9f1465f1fb25e']
config['blacklist'] += ['LKDS-00065', 'LKDS-00150', 'LKDS-00192', 'LKDS-00238', 'LKDS-00319', 'LKDS-00353', 'LKDS-00359',
                        'LKDS-00379', 'LKDS-00504', 'LKDS-00541', 'LKDS-00598', 'LKDS-00684', 'LKDS-00829', 'LKDS-00926']
config['blacklist'] += ['LKDS-00013', 'LKDS-00299', 'LKDS-00314', 'LKDS-00433', 'LKDS-00465', 'LKDS-00602', 'LKDS-00648',
                        'LKDS-00651', 'LKDS-00652', 'LKDS-00778', 'LKDS-00881', 'LKDS-00931', 'LKDS-00960' ]
config['blacklist'] = list(set(config['blacklist']))
config['conf_thresh'] = 0.1  # for inference


class PostRes2d(nn.Module):
    def __init__(self, n_in, n_out, stride = 1):
        super(PostRes2d, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv2d(n_in, n_out, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        return out
class PostRes(nn.Module):
    def __init__(self, n_in, n_out, stride = 1):
        super(PostRes, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm3d(n_out)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm3d(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size = 1, stride = stride),
                nn.BatchNorm3d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        return out

class Rec3(nn.Module):
    def __init__(self, n0, n1, n2, n3, p = 0.0, integrate = True):
        super(Rec3, self).__init__()
        
        self.block01 = nn.Sequential(
            nn.Conv3d(n0, n1, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm3d(n1),
            nn.ReLU(inplace = True),
            nn.Conv3d(n1, n1, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n1))

        self.block11 = nn.Sequential(
            nn.Conv3d(n1, n1, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n1),
            nn.ReLU(inplace = True),
            nn.Conv3d(n1, n1, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n1))
        
        self.block21 = nn.Sequential(
            nn.ConvTranspose3d(n2, n1, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(n1),
            nn.ReLU(inplace = True),
            nn.Conv3d(n1, n1, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n1))
 
        self.block12 = nn.Sequential(
            nn.Conv3d(n1, n2, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm3d(n2),
            nn.ReLU(inplace = True),
            nn.Conv3d(n2, n2, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n2))
        
        self.block22 = nn.Sequential(
            nn.Conv3d(n2, n2, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n2),
            nn.ReLU(inplace = True),
            nn.Conv3d(n2, n2, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n2))
        
        self.block32 = nn.Sequential(
            nn.ConvTranspose3d(n3, n2, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(n2),
            nn.ReLU(inplace = True),
            nn.Conv3d(n2, n2, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n2))
 
        self.block23 = nn.Sequential(
            nn.Conv3d(n2, n3, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm3d(n3),
            nn.ReLU(inplace = True),
            nn.Conv3d(n3, n3, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n3))

        self.block33 = nn.Sequential(
            nn.Conv3d(n3, n3, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n3),
            nn.ReLU(inplace = True),
            nn.Conv3d(n3, n3, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n3))

        self.relu = nn.ReLU(inplace = True)
        self.p = p
        self.integrate = integrate

    def forward(self, x0, x1, x2, x3):
        if self.p > 0 and self.training:
            coef = torch.bernoulli((1.0 - self.p) * torch.ones(8))
            out1 = coef[0] * self.block01(x0) + coef[1] * self.block11(x1) + coef[2] * self.block21(x2)
            out2 = coef[3] * self.block12(x1) + coef[4] * self.block22(x2) + coef[5] * self.block32(x3)
            out3 = coef[6] * self.block23(x2) + coef[7] * self.block33(x3)
        else:
            out1 = (1 - self.p) * (self.block01(x0) + self.block11(x1) + self.block21(x2))
            out2 = (1 - self.p) * (self.block12(x1) + self.block22(x2) + self.block32(x3))
            out3 = (1 - self.p) * (self.block23(x2) + self.block33(x3))

        if self.integrate:
            out1 += x1
            out2 += x2
            out3 += x3

        return x0, self.relu(out1), self.relu(out2), self.relu(out3)


class Net(nn.Module):
    def __init__(self, output_feature=False):
        super(Net, self).__init__()
        # The first few layers consumes the most memory, so use simple convolution to save memory.
        # Call these layers preBlock, i.e., before the residual blocks of later layers.
        self.output_feature = output_feature
        self.preBlock = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace = True),
            nn.Conv3d(24, 24, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace = True))
        
        # 3 poolings, each pooling downsamples the feature map by a factor 2.
        # 3 groups of blocks. The first block of each group has one pooling.
        num_blocks_forw = [2,2,3,3]
        num_blocks_back = [3,3]
        self.featureNum_forw = [24,32,64,64,64]
        self.featureNum_back =    [128,64,64]
        for i in range(len(num_blocks_forw)):
            blocks = []
            for j in range(num_blocks_forw[i]):
                if j == 0:
                    blocks.append(PostRes(self.featureNum_forw[i], self.featureNum_forw[i+1]))
                else:
                    blocks.append(PostRes(self.featureNum_forw[i+1], self.featureNum_forw[i+1]))
            setattr(self, 'forw' + str(i + 1), nn.Sequential(*blocks))

            
        for i in range(len(num_blocks_back)):
            blocks = []
            for j in range(num_blocks_back[i]):
                if j == 0:
                    if i==0:
                        addition = 3
                    else:
                        addition = 0
                    blocks.append(PostRes(self.featureNum_back[i+1]+self.featureNum_forw[i+2]+addition, self.featureNum_back[i]))
                else:
                    blocks.append(PostRes(self.featureNum_back[i], self.featureNum_back[i]))
            setattr(self, 'back' + str(i + 2), nn.Sequential(*blocks))

        self.maxpool1 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.unmaxpool1 = nn.MaxUnpool3d(kernel_size=2,stride=2)
        self.unmaxpool2 = nn.MaxUnpool3d(kernel_size=2,stride=2)

        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace = True))
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace = True))
        self.drop = nn.Dropout3d(p = 0.5, inplace = False)
        self.output = nn.Sequential(nn.Conv3d(self.featureNum_back[0], 64, kernel_size = 1),
                                    nn.ReLU(),
                                    #nn.Dropout3d(p = 0.3),
                                   nn.Conv3d(64, 5 * len(config['anchors']), kernel_size = 1))

    def forward(self, x, coord):
        out = self.preBlock(x)#16
        out_pool,indices0 = self.maxpool1(out)
        out1 = self.forw1(out_pool)#32
        out1_pool,indices1 = self.maxpool2(out1)
        out2 = self.forw2(out1_pool)#64
        #out2 = self.drop(out2)
        out2_pool,indices2 = self.maxpool3(out2)
        out3 = self.forw3(out2_pool)#96
        out3_pool,indices3 = self.maxpool4(out3)
        out4 = self.forw4(out3_pool)#96
        #out4 = self.drop(out4)
        
        rev3 = self.path1(out4)
        comb3 = self.back3(torch.cat((rev3, out3), 1))#96+96
        #comb3 = self.drop(comb3)
        rev2 = self.path2(comb3)
        
        feat = self.back2(torch.cat((rev2, out2,coord), 1))#64+64
        comb2 = self.drop(feat)
        out = self.output(comb2)
        size = out.size()
        out = out.view(out.size(0), out.size(1), -1)
        #out = out.transpose(1, 4).transpose(1, 2).transpose(2, 3).contiguous()
        out = out.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 5)
        out[:,:,:,:,:,0] = torch.sigmoid(out[:,:,:,:,:,0])  # modify loss.py -> BCELoss
        #out = out.view(-1, 5)

        if self.output_feature:
            return feat, out
        else:
            del feat
            return out, x
        

    
def get_model(output_feature=False):
    net = Net(output_feature)
    loss = Loss(config['num_hard'], recon_loss_scale=0.0, class_loss='BCELoss', average=True)
    get_pbb = GetPBB(config)
    return config, net, loss, get_pbb
