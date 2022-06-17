#!/usr/bin/env python
'''
Usage: python run.py im1Name im2Name prePred frame_idx
This script is more complex, we have to give the frame number adn the residual
Assume that the ppredicted frame is of name "123p.png", and the reference frame is of name '123r.png'
For the h and c, I will just name it 'h.npy' and 'c.npy'
'''
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import torch
from torch.autograd import Variable as var
# import numpy as np
import sepconv
import torch.nn.functional as func
import torch.nn as nn
import cv2
from model_LD_opt_head_res import Network
from multiprocessing.connection import Listener
# from Opt import Opter
import time
from IPython import embed
import os
import sys
##########################################################

assert(int(torch.__version__.replace('.', '')) >= 40) # requires at least pytorch version 0.4.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

net = Network().eval().cuda()

net.load_my_state_dict(torch.load('D:\PKU\VVC\VVCSoftware_VTM-VTM-3.0\VVCSoftware_VTM-VTM-3.0\mini_pack\LD_model_iter033-ltype_fSATD_fs-lr_0.0005-trainloss_0.5763-evalloss_0.1075-evalpsnr_30.7258.pkl'))

def do_prediction(img1str, img2str, img3str, outpath, frame_idx, cur_path):
    # 3/4 LD
    # 1/2 RA
    
    #print(sys.argv)    
    # img1str = sys.argv[1]
    # img2str = sys.argv[2]
    # img3str = sys.argv[3]
    # img4str = sys.argv[4]
    # E.g. '123r.png'

    '''
    img?str should be the input image file path
    '''
    # print("!", frame_idx)
    with torch.no_grad():
        frame_idx = int(frame_idx)
        img1 = cv2.imread(img1str)[:,:,::-1]
        img2 = cv2.imread(img2str)[:,:,::-1]
        if frame_idx == 2:
            img3 = img2
        else:
            img3 = cv2.imread(img3str)[:,:,::-1]
        #imgpred = cv2.imread(imgpredstr)[:,:,::-1]
        

        
        
        height = img1.shape[0]
        width  = img1.shape[1]

        img1 = var(torch.from_numpy(img1.transpose(2,0,1).astype('float32') / 255.0)).view(1, 3, height, width).cuda()
        img2 = var(torch.from_numpy(img2.transpose(2,0,1).astype('float32') / 255.0)).view(1, 3, height, width).cuda()
        img3 = var(torch.from_numpy(img3.transpose(2,0,1).astype('float32') / 255.0)).view(1, 3, height, width).cuda()


        intpaddingwidth = 0
        intpaddingheight = 0

        inthei = height
        intwid = width
        if inthei != (inthei >> 5) << 5:
            intpaddingheight = (((inthei >> 5) + 1) << 5) - inthei
        if intwid != (intwid >> 5) << 5:
            intpaddingwidth = (((intwid >> 5) + 1) << 5) - intwid
            # exit()
        # IPython.embed()
        intpaddingleft = int(intpaddingwidth/2)
        intpaddingright = int(intpaddingwidth/2)
        intpaddingtop = int(intpaddingheight/2)
        intpaddingbottom = int(intpaddingheight/2)

        intpaddingleft = int(intpaddingwidth/2)
        intpaddingright = int(intpaddingwidth/2)
        intpaddingtop = int(intpaddingheight/2)
        intpaddingbottom = int(intpaddingheight/2)

        modulePaddingInput = torch.nn.Sequential()
        modulePaddingOutput = torch.nn.Sequential()

        modulePaddingInput = torch.nn.ReplicationPad2d(padding=[intpaddingleft, intpaddingright, intpaddingtop, intpaddingbottom])
        modulePaddingOutput = torch.nn.ReplicationPad2d(padding=[0 - intpaddingleft, 0 - intpaddingright, 0 - intpaddingtop,0 - intpaddingbottom])

        img1 = modulePaddingInput(img1)
        img2 = modulePaddingInput(img2)
        img3 = modulePaddingInput(img3)

        res = img2 - img3


        h_p= os.path.join(cur_path, 'h.npy')
        c_p= os.path.join(cur_path, 'c.npy')

        if frame_idx == 2:
            output, stat = net(img1, img2, do_MA=True)
        else:
            # print("!", frame_idx, type(frame_idx))
            
            state_h = torch.from_numpy(np.load(h_p)).cuda()
            state_c = torch.from_numpy(np.load(c_p)).cuda()
            stat = (state_h, state_c)
            output, stat = net(img1, img2, res, stat, do_MA=True)

        output = modulePaddingOutput(output)
        output_np = ((output[0].cpu().numpy().clip(0.0, 1.0).transpose(1,2,0) * 255.0).astype('uint8'))[:,:,::-1]
        cv2.imwrite(outpath, output_np)


        (state_h, state_c) = stat
        np.save(h_p, state_h.cpu().numpy())
        np.save(c_p, state_c.cpu().numpy())


if __name__ == '__main__':
    print(sys.argv)
    do_prediction(*sys.argv[1:])
