"""
Network architecture for v9 noCR

In this code, we assume the keyframe branch and the flow estimation network are pretrained.
We have provided the pretrained models used in our experiments. You can also use models other than
what we provided.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from collections import OrderedDict

from models.archs import arch_util

#### keyframe branches
from models.archs.colorizers.siggraph17 import siggraph17
from models.archs.colorizers.eccv16 import eccv16


#### Flow estimation
from models.archs.networks.FlowNet2 import FlowNet2
from models.archs.flow_vis import *
from models.archs.networks.resample2d_package.resample2d import Resample2d

class WeightingNet(nn.Module):
    def __init__(self, input=352, output=1):
        super(WeightingNet, self).__init__()
        self.conv1 = nn.Conv2d(input, 196, 3, padding=1)
        self.conv2 = nn.Conv2d(196, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, output, 3, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        x = self.conv3(x)
        return x

class Feature_Refine(nn.Module):
    def __init__(self, input=288, output=128):
        super(Feature_Refine, self).__init__()
        self.conv1 = nn.Conv2d(input, 196, 3, padding=1)
        self.conv2 = nn.Conv2d(196, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, output, 3, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        x = self.conv3(x)
        return x
        
class Channel_Reduction(nn.Module):
    def __init__(self, input, output):
        super(Channel_Reduction, self).__init__()
        self.conv1 = nn.Conv2d(input, output, 1, padding=0)
        self.conv2 = nn.Conv2d(output, output, 3, padding=1)
        self.conv3 = nn.Conv2d(output, output, 1, padding=0)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        x = self.conv3(x)
        return x
        
class Channel_Reduction_1x1(nn.Module):
    def __init__(self, input, output):
        super(Channel_Reduction_1x1, self).__init__()
        self.conv1 = nn.Conv2d(input, output, 1, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x

class flownet_options():
    def __init__(self):
        super(flownet_options, self).__init__()
        self.rgb_max = 1.0
        self.fp16 = False



class TCVC_IDC(nn.Module):
    def __init__(
        self, nf=64, N_RBs=2, key_net="sig17", dataset="DAVIS", train_flow_keyNet=False
    ):
        super(TCVC_IDC, self).__init__()

        self.key_net = key_net
        
        self.L_fea_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=True)        
        )
        
        #self.channel_reduction = Channel_Reduction_1x1(128, 64).cuda()
        self.channel_reduction = Channel_Reduction(128, 64).cuda()
        
        
        self.weigting = WeightingNet(32*3+128*2, 1).cuda()
        self.feature_refine = Feature_Refine(32*3+64*3, 128).cuda()
        
        
        self.need_conv = False
        

        #### define keyframe branch
        if key_net == "sig17":
            self.fea_key = siggraph17(pretrained=True, model_dir="../experiments/pretrained_models")
            nf_key = 128
        elif key_net == "eccv16":
            self.fea_key = eccv16(pretrained=True, model_dir="../experiments/pretrained_models")
            nf_key = 313
        else:
            raise NotImplementedError("Currently only support Sig17")
            
        self.fea_key.eval()
            

        #### SPyNet for flow warping
#         self.flow = spynet.SpyNet()
        
        opts = flownet_options()
        self.flow = FlowNet2(opts)
        self.flow.load_state_dict(torch.load("../experiments/pretrained_models/FlowNet2_checkpoint.pth.tar")['state_dict'])
        self.flow.eval()
        
        self.flow_warping = Resample2d()
        
        
        self.sigmoid = nn.Sigmoid()
        self.MSE = nn.L1Loss(size_average=True)
        
        ### for training only the interframe branch
        if not train_flow_keyNet:
            #for name, param in self.fea_key.named_parameters():
            #     if 'model_out' in name:
            #         param.requires_grad = False
            for param in self.fea_key.parameters():
                param.requires_grad = False
            for param in self.flow.parameters():
                param.requires_grad = False
                

    def forward(self, x, first_key_HR=None, first_key_fea=None):
        """Assuming M + 2 frames with keyframes at two end points
        input:
            x: LR frames
                - [(B, N, nf, H, W), (B, N, nf, H, W), ..., (B, N, nf, H, W), (B, N, nf, H, W)]
                - In total M + 2 entries
                - N: depends on the keyframe branch
            first_key_HR: HR output of the first keyframe - (B, 3, H, W)
            first_key_fea: features of the first keyframe for forward prop. - (B, nf, H, W)
        output:
            out: output HR frames - (B, N + 1, 3, H, W)
            last_key_HR: HR output of the last keyframe - (B, 3, H, W)
            fea_backward_output: features of the last keyframe - (B, nf, H, W)
        """

        B, C, H, W = x[0].size()  # N frames, C = 3
        '''
        if self.training:
            self.fea_key.train()
        else:
            self.fea_key.eval()
        '''
        self.fea_key.eval()
        
        # first key frame
        x_p = x[0]
        if first_key_fea is not None and first_key_HR is not None:
            key_p_HR, fea_forward = first_key_HR, first_key_fea
        else:
            key_p_HR, fea_forward = self.fea_key(x_p)
        out_l = []
        out_l.append(key_p_HR)

        # last key frame
        x_n = x[-1]
        last_key_HR, fea_backward = self.fea_key(x_n)
        
        #### backward propagation
        backward_fea_l = []
        backward_fea_l.insert(0, fea_backward)
        for i in range(len(x) - 2, 0, -1):
            x_n = x[i+1]
            x_current = x[i]
            
            
            flow = self.flow(x_current.repeat(1, 3, 1, 1)+0.5, x_n.repeat(1, 3, 1, 1)+0.5)
            #fea_backward = arch_util.flow_warp(fea_backward, flow.permute(0, 2, 3, 1))    # init
            fea_backward = self.flow_warping(fea_backward, flow)

            input_ = fea_backward

            fea_backward = input_ 

            backward_fea_l.insert(0, fea_backward)
        
            
        #### forward propagation
        ab_fwarp_l = []
        non_mask_fwarp_l = []
        
        x_current = x[0]
        x_n = x[1]
        flow_n_c = self.flow(x_n.repeat(1, 3, 1, 1)+0.5, x_current.repeat(1, 3, 1, 1)+0.5)
        
        warp_x_c = self.flow_warping(x_current, flow_n_c)
        ab_fwarp = self.flow_warping(key_p_HR, flow_n_c) # [B, 2, H, W]
        
        ab_fwarp_l.append(ab_fwarp)
        non_mask = torch.exp( -50 * torch.sum(x_n - warp_x_c, dim=1).pow(2) ).unsqueeze(1)            
        non_mask_fwarp_l.append(non_mask)
        
        forward_fea_l = []
        forward_fea_l.append(fea_forward)
        for i in range(1, len(x) - 1):
            x_p = x[i-1]
            x_current = x[i]
            x_n = x[i+1]
            
            x_p_fea = self.L_fea_extractor(x_p)
            x_c_fea = self.L_fea_extractor(x_current)
            x_n_fea = self.L_fea_extractor(x_n)
            

            flow = self.flow(x_current.repeat(1, 3, 1, 1)+0.5, x_p.repeat(1, 3, 1, 1)+0.5)
            fea_forward = self.flow_warping(fea_forward, flow)
            
            # weighting network
            W = torch.sigmoid(self.weigting(torch.cat([x_p_fea, x_c_fea, x_n_fea, backward_fea_l[i - 1], fea_forward], dim=1)))  # [B, 1, H, W]   in_c: 32*3+128+128
            input_ =  W * fea_forward +  (1 - W) * backward_fea_l[i - 1]  # [B, 128, H, W]
            
            # feature refine network
            x_n_backward_fea = self.channel_reduction(backward_fea_l[i].detach())
            x_p_forward_fea = self.channel_reduction(forward_fea_l[i-1].detach())
            x_c_fusion_fea = self.channel_reduction(input_.detach())
            fea_residual = self.feature_refine(torch.cat([x_p_fea, x_c_fea, x_n_fea, x_n_backward_fea, x_p_forward_fea, x_c_fusion_fea], dim=1))  # in_c: 32*3 + 64*3
            
            fea_forward = input_ + fea_residual
            
            
            # color refine network
            out = self.fea_key.model_out(fea_forward)
            
            out_l.append(out)
            
            flow_n_c = self.flow(x_n.repeat(1, 3, 1, 1)+0.5, x_current.repeat(1, 3, 1, 1)+0.5)            
            warp_x_c = self.flow_warping(x_current, flow_n_c)
            ab_fwarp = self.flow_warping(out, flow_n_c) # [B, 2, H, W]
            
            
            ab_fwarp_l.append(ab_fwarp)
            
            non_mask = torch.exp( -50 * torch.sum(x_n - warp_x_c, dim=1).pow(2) ).unsqueeze(1)            
            non_mask_fwarp_l.append(non_mask)
            
            forward_fea_l.append(fea_forward)

        out_l.append(last_key_HR)
        out = torch.stack(out_l, dim=1)
        ab_fwarp_l_stack = torch.stack(ab_fwarp_l, dim=1)  # [B, N-2, H, W]
        non_mask_fwarp_l_stack = torch.stack(non_mask_fwarp_l, dim=1)  # [B, N-2, H, W]
        
        # warp loss with t=2            
        if len(x) >= 3:
            warp_t = 2
            ab_fwarp_l_2 = []
            non_mask_fwarp_l_2 = []
            for i in range(0, len(x) - warp_t, 1):
                x_current = x[i]
                x_n = x[i+warp_t]
                flow_n_c = self.flow(x_n.repeat(1, 3, 1, 1)+0.5, x_current.repeat(1, 3, 1, 1)+0.5)            
                warp_x_c = self.flow_warping(x_current, flow_n_c)
                ab_fwarp = self.flow_warping(out_l[i], flow_n_c) # [B, 2, H, W]
                ab_fwarp_l_2.append(ab_fwarp)
                
                non_mask = torch.exp( -50 * torch.sum(x_n - warp_x_c, dim=1).pow(2) ).unsqueeze(1)            
                non_mask_fwarp_l_2.append(non_mask)
                
            ab_fwarp_l_stack_2 = torch.stack(ab_fwarp_l_2, dim=1)  # [B, N-2, H, W]
            non_mask_fwarp_l_stack_2 = torch.stack(non_mask_fwarp_l_2, dim=1)  # [B, N-2, H, W]
        else:
            ab_fwarp_l_stack_2 = None
            non_mask_fwarp_l_stack_2 = None
        
        
        return out, ab_fwarp_l_stack, non_mask_fwarp_l_stack, ab_fwarp_l_stack_2, non_mask_fwarp_l_stack_2
