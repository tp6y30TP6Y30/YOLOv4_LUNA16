'''
Convolution Capsule linear network
'''
#!/usr/bin/python3
#coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GetPBB
from loss import Loss_recon
from ConvCapsule_layers_3d import ConvCapsuleLayer as ConvCap
from ConvCapsule_layers_3d import ConvertToCapsule, FlatCap, CapToLength
from ConvCapsule_layers_3d import ConvCapResidualBlock as ConvCapBlock
from ConvCapsule_layers_3d import DenseBlock as DB




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



class CapNet(nn.Module):
    def __init__(self, output_feature=False):
        super().__init__()
        self.debug = False
        self.output_feature = output_feature

        self.flatcap = FlatCap()
        self.conv_to_cap0 = ConvertToCapsule(num_cap=1)
        self.conv_to_cap1 = ConvertToCapsule(num_cap=1)
        self.cap_len = CapToLength(keepdim=False)

        self.conv0_1 = nn.Conv3d(1, 64, 3, 1, 1)
        self.n0_1 = nn.InstanceNorm3d(64)
        self.conv0_2 = ConvCap(64, 8, 8, 3, 1, 1, 1, 1, 1)
        self.conv1 = ConvCapBlock(8, 8, 4, 16, 3, 2, 1, 1, 3)
        self.block1 = DB(3, 4, 16, 4)
        self.conv2 = ConvCapBlock(16, 16, 4, 32, 3, 2, 1, 1, 3)
        self.block2 = DB(3, 4, 32, 4)

        self.class_head = ConvCap(input_num_atoms=32 + 3, num_capsule=len(config['anchors']), num_atoms=16,
                                  kernel_size=3, strides=1, padding=1, dilation=1, groups=1, routings=3)
        self.loc_head = ConvCap(input_num_atoms=16, num_capsule=len(config['anchors']), num_atoms=4,
                                kernel_size=3, strides=1, padding=1, dilation=1, groups=1, routings=3)
        # self.recon = nn.Sequential(
        #     FlatCap(),
        #     nn.ReLU(),
        #     nn.ConvTranspose3d(len(config['anchors']) * 16, 128, 2, 2, 0),
        #     nn.ReLU(),
        #     nn.ConvTranspose3d(128, 1, 2, 2, 0)
        # )

    def forward(self, x, coord):
        if self.debug: print(f'x={x.size()} coord={coord.size()}')

        out0_1 = F.relu(self.n0_1(self.conv0_1(x)))
        if self.debug: print(f'out0_1={out0_1.size()}')

        out0_2 = self.conv0_2(self.conv_to_cap0(out0_1))
        if self.debug: print(f'out0_2={out0_2.size()}')

        out1 = self.conv1(out0_2)
        if self.debug: print(f'out1={out1.size()}')

        out1 = self.block1(out1)
        if self.debug: print(f'block1={out1.size()}')

        out2 = self.conv2(out1)
        if self.debug: print(f'out2={out2.size()}')

        out2 = self.block2(out2)
        if self.debug: print(f'block2={out2.size()}')

        feature_maps = out2
        if self.debug: print(f'feature_maps={feature_maps.size()}')

        # ============ Compute location and classification ============
        n_cap = feature_maps.size(4)
        coord = self.conv_to_cap1(coord).repeat(1, 1, 1, 1, n_cap, 1)
        if self.debug: print(f'coord={coord.size()}')

        comb = torch.cat((feature_maps, coord), -1)
        if self.debug: print(f'comb={comb.size()}')

        feat = self.flatcap(comb)
        if self.debug: print(f'feat={feat.size()}')

        class_cap = self.class_head(comb)
        if self.debug: print(f'class_cap={class_cap.size()}')

        class_len = self.cap_len(class_cap)
        if self.debug: print(f'class_len={class_len.size()}')

        loc_cap = self.loc_head(class_cap)
        if self.debug: print(f'loc_cap={loc_cap.size()}')

        out = torch.cat((class_len.unsqueeze(5), loc_cap), 5)
        if self.debug: print(f'class_len+loc_cap={out.size()}')

        # ============ Reconstruct image by all capsules ============
        # reconstruction = self.recon(class_cap)
        # if self.debug: print(f'reconstruction={reconstruction.size()}')


        # ============ Recycle unused variables ============
        del out0_1, out0_2
        del out1
        del out2
        del feature_maps, comb, class_cap, class_len, loc_cap


        if self.output_feature:
            # return feat, out, reconstruction
            return feat, out, x
        else:
            # return out, reconstruction
            return out, x


def get_model(output_feature=False):
    net = CapNet(output_feature)
    loss = Loss_recon(config['num_hard'], recon_loss_scale=0.0, class_loss='MarginLoss', average=False)
    get_pbb = GetPBB(config)
    return config, net, loss, get_pbb

def test():
    device = torch.device('cuda')
    net = CapNet().to(device)
    n = 1
    x = torch.randn(n, 1, 48, 48, 48).to(device)
    crd = torch.randn(n, 3, 12, 12, 12).to(device)

    total_params = sum(p.numel() for p in net.parameters())
    print(f'Total number of parameters is {total_params}')

    if net.output_feature:
        feat, out, recon = net(x, crd)
        print(f'feat={feat.size()}  out={out.size()} recon={recon.size()}')
    else:
        out, recon = net(x, crd)
        print(f'out={out.size()}  recon={recon.size()}')



if __name__=='__main__':
    test()
