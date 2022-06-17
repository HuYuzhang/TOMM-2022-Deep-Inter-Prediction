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
from model_RA_opt_long_inject import Network
from multiprocessing.connection import Listener
from Opt import Opter
import time
from IPython import embed
##########################################################

assert(int(torch.__version__.replace('.', '')) >= 40) # requires at least pytorch version 0.4.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

net = Network().eval().cuda()
# net.load_state_dict(torch.load('/home/struct/TIPtest/code/RA_model_iter004-ltype_fSATD_fs-lr_0.0001-trainloss_0.4861-evalloss_0.0924-evalpsnr_32.864.pkl'))
# net.load_state_dict(torch.load('/home/struct/TIPtest/code/RA_model_iter007-ltype_fSATD_fs-lr_0.0005-trainloss_0.0957-evalloss_0.0912-evalpsnr_33.0997.pkl'))
#net.load_state_dict(torch.load('/home/struct/TIPtest/code/RA_model_iter020-ltype_fSATD_fs-lr_0.0005-trainloss_0.0952-evalloss_0.0909-evalpsnr_33.128.pkl'))
#net.load_state_dict(torch.load('/home/struct/TIPtest/code/RA_model_iter024-ltype_fSATD_fs-lr_0.0005-trainloss_0.0934-evalloss_0.0892-evalpsnr_33.3738.pkl'))
#net.load_state_dict(torch.load('/home/struct/TIPtest/code/RA_model_iter031-ltype_fSATD_fs-lr_0.0005-trainloss_0.0933-evalloss_0.0891-evalpsnr_33.4009.pkl'))
#net.load_state_dict(torch.load('/home/struct/TIPtest/code/RA_model_iter010-ltype_fSATD_fs-lr_0.0005-trainloss_0.5311-evalloss_0.1003-evalpsnr_31.788.pkl'))
#net.load_state_dict(torch.load('/home/struct/TIPtest/code/RA_model_iter062-ltype_fSATD_fs-lr_0.0005-trainloss_0.0933-evalloss_0.089-evalpsnr_33.4092.pkl'))
#net.load_state_dict(torch.load('/home/struct/TIPtest/code/RA_model_iter023-ltype_fSATD_fs-lr_0.0005-trainloss_0.4832-evalloss_0.0924-evalpsnr_32.9115.pkl'))
#net.load_state_dict(torch.load('/home/struct/TIPtest/code/RA_model_iter023-ltype_fSATD_fs-lr_0.0005-trainloss_0.5216-evalloss_0.0986-evalpsnr_32.0118.pkl'))
#net.load_my_state_dict(torch.load('D:\PKU\VVC\code\model.pkl'))
net.load_my_state_dict(torch.load('RA_model_iter043-ltype_fSATD_fs-lr_0.0005-trainloss_0.97-evalloss_0.0897-evalpsnr_33.2704.pkl')) # the pkl on the formal version [*]
# net.load_my_state_dict(torch.load('vimeo_vvc_epoch14.pkl')) 

opt = Opter()


def do_prediction(img1str, img2str, img3str, outpath, tagpath):
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
    # embed()
    # exit()

    with torch.no_grad():
        img1 = cv2.imread(img1str)[:,:,::-1]
        img2 = cv2.imread(img2str)[:,:,::-1]
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

        flo1 = opt.estimate(img1, img3)
        flo2 = opt.estimate(img2, img3)

        tensorLong = torch.cat([flo1, flo2], axis=1)
        #tensorLong = torch.zeros([1, 4, flo1.shape[2], flo1.shape[3]]).cuda()#torch.cat([flo1, flo2], axis=1)

        output = net(img1, img2, tensorLong, do_MA=False)


        flo = opt.estimate(output, img3)
        warped_raw = opt.warp(img3, flo)
        warped = (warped_raw-output)
        output = net(output, warped, only_MA=True)

        output = modulePaddingOutput(output)
        output_np = ((output[0].cpu().numpy().clip(0.0, 1.0).transpose(1,2,0) * 255.0).astype('uint8'))[:,:,::-1]
        cv2.imwrite(outpath, output_np)

        f = open(tagpath, 'w')
        f.close()
    

if __name__ == '__main__':
    port = 6000
    address = ('localhost', port)
    listener = Listener(address, authkey=b'secret password')
    print('begin to listen')
    while True:
        try:
            conn = listener.accept()
            print('accept from ', listener.last_accepted)
            msg = conn.recv()
            msg = msg.split(' ')[1:]
            print (msg)
            # print('receive: ', msg)
        # conn.send('hello')
        # msg = conn.recv_bytes(maxlength=5)
        # msg = str(msg)
        # print('receive: ', msg)
        # msg = msg.split(' ') # img1, img2, img3, img4, outimg
            # embed()
            do_prediction(*msg)
        except:
            print("error occcured")
            embed()
            exit()
        # break
    # conn.send(b'finish')
    # conn.close()

listener.close()
