from PIL import Image
import numpy
import math

def calcPSNR(im1, im2):
    bsize,c,height,width = im1.shape
    psnrsum=0.0
    for i in range(0,bsize):
        R = im1[i,0,:,:]-im2[i,0,:,:]
        G = im1[i,1,:,:]-im2[i,1,:,:]
        B = im1[i,2,:,:]-im2[i,2,:,:]
        mser = R*R
        mseg = G*G
        mseb = B*B
        SUM = mser.sum() + mseg.sum() + mseb.sum()
        MSE = SUM / (height * width * 3)
        PSNR = 10*math.log ( (1/(MSE)) ,10)
        psnrsum = psnrsum + PSNR
    return psnrsum / bsize

def calcPSNRnp(im1, im2):
    height,width,c = im1.shape
    psnrsum=0.0
    R = im1[:,:,0]-im2[:,:,0]
    G = im1[:,:,1]-im2[:,:,1]
    B = im1[:,:,2]-im2[:,:,2]
    mser = R*R
    mseg = G*G
    mseb = B*B
    SUM = mser.sum() + mseg.sum() + mseb.sum()
    MSE = SUM / (height * width * 3)
    PSNR = 10*math.log ( (1/(MSE + 0.0000000000001)) ,10)
    psnrsum = psnrsum + PSNR
    return psnrsum
