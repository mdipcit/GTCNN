# -*- coding: utf-8 -*-

# =============================================================================
# Based on the code from https://github.com/cszn/DnCNN/tree/master/TrainingCodes/dncnn_pytorch (by Kai Zhang)
# =============================================================================



import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import torchvision

class DenoisingDataset(Dataset):

    def __init__(self, xs, sigma,img_size=180,random_corp=False):
        super(DenoisingDataset, self).__init__()
        assert sigma<=100 and sigma>=10, 'Sigma was expected between with 10 to 100, but got [{0}]'.format(sigma)
        self.xs = xs
        self.sigma = sigma
        self.img_size =img_size
        self.random_corp = random_corp
    def __getitem__(self, index):
        if self.random_corp:
            batch_x = self.get_patch(self.xs[index]) # augmantation 
            noise = torch.randn(batch_x.size()).mul_(self.sigma/255.0)
            batch_y = batch_x + noise
        else:
            batch_x = self.xs[index]
            noise = torch.randn(batch_x.size()).mul_(self.sigma/255.0)
            batch_y = batch_x + noise
        return batch_y, batch_x

    def get_patch(self,img_in):
        _,ih, iw = img_in.size()
        ix = torch.randint(0,iw - self.img_size + 1,(1,)).item()
        iy = torch.randint(0,ih - self.img_size + 1,(1,)).item()
        def _augment(img):
            hflip =  np.random.random() < 0.5
            vflip =  np.random.random() < 0.5
            if hflip: img = torch.flip(img,(2,))
            if vflip: img =  torch.flip(img,(1,))
            return img
        img_in = img_in[:,iy:iy + self.img_size, ix:ix + self.img_size]

        return _augment(img_in)


    def collate_fn(self, batch):
        batch_y, batch_x = list(zip(*batch))
        batch_x = torch.stack([img for img in batch_x])
        batch_y = torch.stack([img for img in batch_y])
        return batch_y, batch_x
    def __len__(self):
        return self.xs.size(0)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".JPEG"])


def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def set_image(img,size):# Crop image to arange image size
    if img.shape[0] != img.shape[1]:
        if img.shape[0] <img.shape[1]:
            y,x ,_= img.shape
            x= y
            img= img[:,0:x,:]
            width =img.shape[0]
        else:
            y,x ,_ = img.shape
            y=x
            img= img[0:y,:,:]
            width =img.shape[1]
    else: width =img.shape[0]
    if size is not None:

        img = crop_center(size,size)
        if len(img.shape)<3:
            img = np.expand_dims(img, axis=2)

    return img




def gen_patches(file_name,patch_size,patch_crop,stride,large_size=False,scales=[1],color=1):
    if color == 1:
        img = cv2.imread(file_name, 0)  # gray scale
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif color == 3:
        img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)  # BGR or G
    if large_size:
        patch_size=large_size
    patches = []
    if patch_crop==True:
        w, h,_ = img.shape
        for i in range(0, w-patch_size+1, stride):
            for j in range(0, h-patch_size+1, stride):
                x = img[i:i+patch_size, j:j+patch_size,:]
                patches.append(x)

    else:
        patches.append(set_image(img,patch_size))
    return patches


def nsat_datagenerator(root = "dataset/",classes = ['mix'], verbose=False, batch_size=32, patch_size=63,
            Nsat=400,patch_crop=False,large_size=False, stride = 100,scales=[1],color=1):
    # generate clean patches from a dataset
    data = []
    for index,c in enumerate(classes):
        # index=1
        file_list = glob.glob(root+c+'/*')  # get name list 
        if Nsat[1] == -1:
            file_list=file_list
        else:
            file_list=file_list[Nsat[0]:Nsat[1]]
        for i in range(len(file_list)):
            assert is_image_file(file_list[i]),"[{0}] is not image file. check your dataset".format(file_list[i])
            patches = gen_patches(file_list[i],patch_size,patch_crop,stride,large_size=large_size,scales=scales,color=color)
            for patch in patches:    

                data.append(patch)


    data = np.array(data, dtype='uint8')

    print(data.shape)



    print('^_^-training data finished-^_^')
    return data

