'''
Dense CNN linear network
LR=0.001
Mini batch = 48
'''
#!/usr/bin/python3
#coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GetPBB
from loss_2 import Loss_recon as Loss



config = {}
config['anchors'] = [5., 10., 20.]  #[ 10.0, 30.0, 60.]
config['channel'] = 1
config['crop_size'] = [80, 80, 80]      #[96, 96, 96]
config['stride'] = 4
config['max_stride'] = 16
config['num_neg'] = 800
config['th_neg'] = 0.02
config['th_pos_train'] = 0.5
config['th_pos_val'] = 1
config['num_hard'] = 2
config['bound_size'] = 12
config['reso'] = 1
config['sizelim'] = 2.5   #6. #mm
config['sizelim2'] = 10.  #30
config['sizelim3'] = 20.  #40
config['sizelim4'] = 40.
config['aug_scale'] = True
config['r_rand_crop'] = 0.3
config['pad_value'] = 170
config['augtype'] = {'flip':True, 'swap':False, 'scale':True, 'rotate':False}
config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38', '990fbe3f0a1b53878669967b9afd1441', 'adc3bbc63d40f8761c59be10f1e504c3',
                       'f938f9022abf7f1072fe9df79db7eccd', 'eb008af181f3791fdce2376cf4773733', '11fe5426ef497bc490b9f1465f1fb25e']
config['blacklist'] += ['LKDS-00065', 'LKDS-00150', 'LKDS-00192', 'LKDS-00238', 'LKDS-00319', 'LKDS-00353', 'LKDS-00359',
                        'LKDS-00379', 'LKDS-00504', 'LKDS-00541', 'LKDS-00598', 'LKDS-00684', 'LKDS-00829', 'LKDS-00926']
config['blacklist'] += ['LKDS-00013', 'LKDS-00299', 'LKDS-00314', 'LKDS-00433', 'LKDS-00465', 'LKDS-00602', 'LKDS-00648',
                        'LKDS-00651', 'LKDS-00652', 'LKDS-00778', 'LKDS-00881', 'LKDS-00931', 'LKDS-00960' ]
config['blacklist'] = list(set(config['blacklist']))
config['conf_thresh'] = 0.5  # for inference


class _DenseLayer(nn.Module):
    def __init__(self, in_channel, growth_rate, kernel_size=3, drop_rate=0):
        super(_DenseLayer, self).__init__()
        padding = (kernel_size - 1) // 2
        self.norm_1 = nn.BatchNorm3d(in_channel)
        self.relu_1 = nn.ReLU()
        self.conv_1 = nn.Conv3d(in_channel, growth_rate*4, kernel_size=1, stride=1, padding=0) #batch size = 4
        self.norm_2 = nn.BatchNorm3d(growth_rate*4 )#batch size = 4
        self.relu_2 = nn.ReLU()
        self.conv_2 = nn.Conv3d(growth_rate*4, growth_rate, kernel_size=kernel_size, stride=1, padding=padding) #batch size = 4

        self.drop_rate = drop_rate
    def forward(self, x):
        """
        :param x: (n, in_channel, d, h, w)
        :return: (n, in_channel+growth_rate, d, h, w)
        """
        out = self.norm_1(x)
        out = self.relu_1(out)
        out = self.conv_1(out)
        out = self.norm_2(out)
        out = self.relu_2(out)
        out = self.conv_2(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


class Transition(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, padding=0, is_pool=1):
        super(Transition, self).__init__()
        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding)
        self.avgpool = nn.AvgPool3d(kernel_size=2, stride=2)
        self.is_pool = is_pool

    def forward(self, x):
        x = self.conv(x)
        if self.is_pool:
            x = self.avgpool(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channel, growth_rate, kernel_size=3):
        super(DenseBlock, self).__init__()
        self.layers = nn.Sequential()
        for i in range(num_layers):
            layer = _DenseLayer(in_channel + i * growth_rate, growth_rate, kernel_size)
            self.layers.add_module(f'DenseLayer{i+1}', layer)

    def forward(self, x):
        """
        :param x: (n, in_channel, d, h, w)
        :return: (n, in_channel + growth_rate*num_layers, d, h, w)
        """
        out = self.layers(x)
        return out


class DenseNet(nn.Module):
    def __init__(self, output_feature=False):
        super().__init__()
        self.debug = False
        self.output_feature = output_feature

        self.conv0 = nn.Conv3d(1, 32, 3, 1, 1)
        self.n0 = nn.BatchNorm3d(32)
        # self.conv1 = nn.Conv3d(24, 32, 3, 2, 1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.block1 = DenseBlock(6, 32, 32, 3)
        self.tran1 = Transition(224, 64, 1, 1, 0, 1)
        #self.conv2 = nn.Conv3d(224, 64, 1, 1, 0)
        #self.avgpool1 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.block2 = DenseBlock(12, 64, 32, 3) 
        self.tran2 = Transition(448, 224, 1, 1, 0, 0)

        self.drop = nn.Dropout3d(p=0.5, inplace=False)
        self.output = nn.Sequential(nn.Conv3d(227, 64, kernel_size=1),
                                    nn.ReLU(),
                                    nn.Conv3d(64, 5 * len(config['anchors']), kernel_size=1))

    def forward(self, x, coord):
        if self.debug: print(f'x={x.size()} coord={coord.size()}')

        out0 = F.relu(self.n0(self.conv0(x)))
        if self.debug: print(f'out0={out0.size()}')

        # out1 = self.conv1(out0)
        out1 = self.maxpool1(out0)
        if self.debug: print(f'out1={out1.size()}')

        out1 = self.block1(out1)
        if self.debug: print(f'block1={out1.size()}')

        out2 = self.tran1(out1)
        if self.debug: print(f'tran1={out2.size()}')

        out2 = self.block2(out2)
        if self.debug: print(f'block2={out2.size()}')

        out2 = self.tran2(out2)
        if self.debug: print(f'tran2={out2.size()}')

        feat = torch.cat((out2, coord), 1)
        if self.debug: print(f'feat={feat.size()}')

        # ============ Compute location and classification ============
        comb2 = self.drop(feat)
        if self.debug: print(f'comb2={comb2.size()}')

        out = self.output(comb2)
        if self.debug: print(f'self.output(comb2)={out.size()}')

        size = out.size()
        out = out.view(out.size(0), out.size(1), -1)
        if self.debug: print(f'out.view={out.size()}')

        out = out.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 5)
        out[:,:,:,:,:,0] = F.sigmoid(out[:,:,:,:,:,0])  # modify loss.py -> BCELoss
        if self.debug: print(f'out.transpose={out.size()}')

        # ============ Recycle unused variables ============
        del out0
        del out1
        del out2
        del comb2


        if self.output_feature:
            return feat, out
        else:
            del feat
            return out, x


def get_model(output_feature=False):
    net = DenseNet(output_feature)
    loss = Loss(config['num_hard'], recon_loss_scale=0.0, class_loss='BCELoss', average=True)
    get_pbb = GetPBB(config)
    return config, net, loss, get_pbb

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config, net, loss, get_pbb = get_model()
    net = net.to(device)

    n = 1
    x = torch.randn(n, 1, 48, 48, 48).to(device)
    crd = torch.randn(n, 3, 12, 12, 12).to(device)

    total_params = sum(p.numel() for p in net.parameters())
    print(f'Total number of parameters is {total_params}')

    if net.output_feature:
        feat, out = net(x, crd)
        print(f'feat={feat.size()}  out={out.size()}')
    else:
        out = net(x, crd)
        print(f'out={out.size()}')


if __name__=='__main__':
    test()
