#!/usr/bin/env python
'''
Usage: python run.py im1Name im2Name prePred frame_idx
This script is more complex, we have to give the frame number adn the residual
Assume that the ppredicted frame is of name "123p.png", and the reference frame is of name '123r.png'
For the h and c, I will just name it 'h.npy' and 'c.npy'
'''
import sys
import math
# import numpy as np
import os
import warnings
import random
warnings.filterwarnings('ignore')
import time
from torch.autograd import Variable as var
from subprocess import Popen
# import numpy as np
import sepconv
import torch.nn.functional as func
import torch.nn as nn
import cv2
from multiprocessing.connection import Client
from IPython import embed
import shutil
import time

##########################################################
quick_flag = True # For class D, this flag is recommended to set to true for faster time
do_prediction_name = '/home/struct/TIPtest/code/do_prediction_ld.py' # For class C/D, your can set it to ld for faster speed
##########################################################
port = 6000
GOP_size = 16 # The GOP size of current coding configuration
if __name__ == '__main__':
    
    img1str = sys.argv[1]
    img2str = sys.argv[2]
    # tid = int(sys.argv[3]) # for each tid, I will maintain the hidden state, here, only 2, 3, 4 layer will be maintained
    img1idx = int(img1str[:-5])
    img2idx = int(img2str[:-5])

    frame_idx = int((img1idx + img2idx) / 2)
    imgout = '%dp.png'%(frame_idx)


    # shutil.copy(img2str, imgout)
    # exit()

    dis = img2idx - frame_idx

    img_ref_str = '%dr.png'%(frame_idx-2*dis)# Two-candidate strategy
    # img_ref_str = '%dr.png'%(frame_idx//GOP_size*GOP_size)# Always key-frame strategy
    # img_ref_str = img2str

    
    if not os.path.exists(img_ref_str):
        img_ref_str = '%dr.png'%(GOP_size*(frame_idx//GOP_size))
    
    cur_path = os.getcwd()
    # print(img_ref_str)
    if os.path.exists(img_ref_str):
        img1path = os.path.join(cur_path, img1str)
        img2path = os.path.join(cur_path, img2str)
        img3path = os.path.join(cur_path, img_ref_str)
        
        outpath = os.path.join(cur_path, imgout)   
        tagpath = os.path.join(cur_path, 'tag.txt') 
        # print('hello')
        if os.path.exists(tagpath):
            os.remove(tagpath)
        # embed()
        # exit()
        
        address = ('localhost', port)
        conn = Client(address, authkey=b'secret password')
        info = cur_path + ' ' + img1str + ' ' + img2str + ' ' + img_ref_str  + ' ' + imgout + ' ' + 'tag.txt'
        # print(info)
        conn.send(info)
        time.sleep(20)
        conn.close()
        while True:
            if os.path.exists(tagpath):
                break
            
            
        
        # os.remove(tagpath)

    else:
        cv2.imwrite(imgout, cv2.imread(img1str))
        exit(0)
            
