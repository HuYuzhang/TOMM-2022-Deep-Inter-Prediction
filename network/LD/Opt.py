'''
This class has two usages: 
1. Calculate the flow net use the FlyNet
2. Warp a tensor with a given flow
'''
import torch

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
import IPython
import cv2
from torch.nn.functional import grid_sample
import IPython
import torch.nn as nn

class Opter():
    def __init__(self, gpu=0):
        #torch.cuda.set_device(gpu)
        self.Backward_tensorGrid = {}
        def Backward(tensorInput, tensorFlow):
            if str(tensorFlow.size()) not in self.Backward_tensorGrid:
                tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
                tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

                self.Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
            # end

            tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

            return torch.nn.functional.grid_sample(input=tensorInput, grid=(self.Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
        # end

        class Network(torch.nn.Module):
            def __init__(self):
                super(Network, self).__init__()

                class Preprocess(torch.nn.Module):
                    def __init__(self):
                        super(Preprocess, self).__init__()
                    # end

                    def forward(self, tensorInput):
                        tensorBlue = (tensorInput[:, 0:1, :, :] - 0.406) / 0.225
                        tensorGreen = (tensorInput[:, 1:2, :, :] - 0.456) / 0.224
                        tensorRed = (tensorInput[:, 2:3, :, :] - 0.485) / 0.229

                        return torch.cat([ tensorRed, tensorGreen, tensorBlue ], 1)
                    # end
                # end

                class Basic(torch.nn.Module):
                    def __init__(self, intLevel):
                        super(Basic, self).__init__()

                        self.moduleBasic = torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
                            torch.nn.ReLU(inplace=False),
                            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
                            torch.nn.ReLU(inplace=False),
                            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
                            torch.nn.ReLU(inplace=False),
                            torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
                            torch.nn.ReLU(inplace=False),
                            torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
                        )
                    # end

                    def forward(self, tensorInput):
                        return self.moduleBasic(tensorInput)
                    # end
                # end

                self.modulePreprocess = Preprocess()

                self.moduleBasic = torch.nn.ModuleList([ Basic(intLevel) for intLevel in range(6) ])
                try:
                    self.load_state_dict(torch.load('/home/struct/TIPtest/code/network-sintel-final.pytorch'))
                except Exception:
                    self.load_state_dict(torch.load('D:\\hyz\\19Summer\\ISCAS\\Extrapolation\\utils\\network-sintel-final.pytorch'))
            # end

            def forward(self, tensorFirst, tensorSecond):
                tensorFlow = []

                tensorFirst = [ self.modulePreprocess(tensorFirst) ]
                tensorSecond = [ self.modulePreprocess(tensorSecond) ]

                for intLevel in range(5):
                    if tensorFirst[0].size(2) > 32 or tensorFirst[0].size(3) > 32:
                        tensorFirst.insert(0, torch.nn.functional.avg_pool2d(input=tensorFirst[0], kernel_size=2, stride=2, count_include_pad=False))
                        tensorSecond.insert(0, torch.nn.functional.avg_pool2d(input=tensorSecond[0], kernel_size=2, stride=2, count_include_pad=False))
                    # end
                # end

                tensorFlow = tensorFirst[0].new_zeros([ tensorFirst[0].size(0), 2, int(math.floor(tensorFirst[0].size(2) / 2.0)), int(math.floor(tensorFirst[0].size(3) / 2.0)) ])

                for intLevel in range(len(tensorFirst)):
                    tensorUpsampled = torch.nn.functional.interpolate(input=tensorFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

                    if tensorUpsampled.size(2) != tensorFirst[intLevel].size(2): tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[ 0, 0, 0, 1 ], mode='replicate')
                    if tensorUpsampled.size(3) != tensorFirst[intLevel].size(3): tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[ 0, 1, 0, 0 ], mode='replicate')

                    tensorFlow = self.moduleBasic[intLevel](torch.cat([ tensorFirst[intLevel], Backward(tensorInput=tensorSecond[intLevel], tensorFlow=tensorUpsampled), tensorUpsampled ], 1)) + tensorUpsampled
                # end

                return tensorFlow
            # end
        # end
        self.FpyNet = Network().cuda()#= nn.DataParallel(Network().cuda(), device_ids=gpu)
        self.FpyNet.eval()
    
    def estimate(self, tensorFirst, tensorSecond):
        '''
        The input can be Tensor of size: [bcz, c, h, w] or [c, h, w], and I will transfer them to .cuda()
        '''
        tensorFirst = tensorFirst.cuda()
        tensorSecond = tensorSecond.cuda()
        if len(tensorFirst.size()) == 3:
            tensorFirst = tensorFirst.unsqueeze(0)
            tensorSecond = tensorSecond.unsqueeze(0)
        
        intWidth = tensorFirst.size(3)
        intHeight = tensorFirst.size(2)
        batch_size = tensorFirst.size(0)
        # assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
        # assert(intHeight == 416) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

        tensorPreprocessedFirst = tensorFirst.view(batch_size, 3, intHeight, intWidth)
        tensorPreprocessedSecond = tensorSecond.view(batch_size, 3, intHeight, intWidth)

        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

        tensorPreprocessedFirst = torch.nn.functional.interpolate(input=tensorPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
        tensorPreprocessedSecond = torch.nn.functional.interpolate(input=tensorPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

        tensorFlow = torch.nn.functional.interpolate(input=self.FpyNet(tensorPreprocessedFirst, tensorPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

        tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

        return tensorFlow

    def warp(self, tensorInput, tensorFlow):
        '''
        The input can be Tensor of size: [bcz, c, h, w] or [c, h, w], and I will transfer them to .cuda()
        '''
        tensorInput = tensorInput.cuda()
        tensorFlow = tensorFlow.cuda()
        if len(tensorInput.size()) == 3:
            tensorInput = tensorInput.unsqueeze(0)
            tensorFlow = tensorFlow.unsqueeze(0)

        if str(tensorFlow.size()) not in self.Backward_tensorGrid:
            tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
            tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

            self.Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
        # end

        tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

        return torch.nn.functional.grid_sample(input=tensorInput, grid=(self.Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
        # end
