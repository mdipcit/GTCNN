import os
import torch
import torch.nn as nn
import numpy as np
import os
import math
import random
import numpy as np
import torch
import cv2

def pad_to_image(x,mod_size=160):
    #  Pad to image for prosessing the odd iamge size.
    shape = x.shape
    shape_w_s = np.arange(shape[0],shape[0]+1000)  
    shape_h_s = np.arange(shape[1],shape[1]+1000)  
    x_w_pad_size = np.where(np.mod(shape_w_s,mod_size)==0)[0][0]
    x_h_pad_size = np.where(np.mod(shape_h_s,mod_size)==0)[0][0]
    if x_w_pad_size>0:
        x = np.concatenate((x,np.flip(x[-x_w_pad_size:, :],0)), axis=0)
    if x_h_pad_size>0:
        x = np.concatenate((x,np.flip(x[:, -x_h_pad_size:],1)), axis=1)
    return x,x_w_pad_size,x_h_pad_size
def shave_pad(x_,x_w_pad_size,x_h_pad_size):
    # Crop padded image to the original image size
    if x_w_pad_size>0:
        if x_.ndim == 3:
            x_ = x_[:-x_w_pad_size,:,:] 
        else:
            x_ = x_[:-x_w_pad_size,:] 
    if x_h_pad_size>0:
        if x_.ndim == 3:
            x_ = x_[:,:-x_h_pad_size,:] 
        else:
            x_ = x_[:,:-x_h_pad_size] 
    return x_
def save(dict,file):
    import pickle
    f = open(dict+'.binaryfile','wb')
    pickle.dump(file,f)
    f.close

def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for v in model.parameters())/1e6






# we borrowed a part of the following code:
# https://github.com/cszn/KAIR
def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)
def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

# convert single (HxWxC) to 4-dimensional torch tensor
def single2tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)
def uint2single(img):

    return np.float32(img/255.)


def single2uint(img):

    return np.uint8((img.clip(0, 1)*255.).round())

# convert torch tensor to single
def tensor2single(img):
    img = img.data.squeeze().clamp_(0, 1).float().cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))

    return img
# --------------------------------------------
# get uint8 image of size HxWxn_channles (RGB)
# --------------------------------------------
def imread_uint(path, n_channels=3):
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img

'''
# --------------------------------------------
# metric, PSNR and SSIM
# --------------------------------------------
'''


# --------------------------------------------
# PSNR
# --------------------------------------------
def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# --------------------------------------------
# SSIM
# --------------------------------------------
def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()





