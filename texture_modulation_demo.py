#!/usr/bin/env python
# -*- coding: utf-8 -*-


#    Copyright 2020 Kaito Imai

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import numpy as np
import time
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import utils
import os, time, datetime
from os import listdir
import glob
import sys
import cv2
from IPython.display import display
from PIL import Image
from models import *

#----------------------------------------------------------------------------------------------------------------
#-----------------------------------main process--------------------------------------
#----------------------------------------------------------------------------------------------------------------

class TM_Demo():
    def __init__(self,iamge_path,hyper_params):
        self.Experiment_Confs = hyper_params
        self.Model_conf =  hyper_params.model
        self.Data_conf =  hyper_params.dataset
        self.hyper_params = hyper_params.experiment
        self.handle_test_size =  self.hyper_params['handle_test_size']
        self.test_mod_size = self.hyper_params['test_mod_size']
        self.model_save_dir = self.hyper_params['model_save_dir']
        self.gpuID=self.hyper_params['device_ids'][0]
        self.iamge_path=iamge_path
        self.model = eval(self.Model_conf.name)(self.Model_conf)

        state_dict = torch.load(self.model_save_dir+self.hyper_params["best_model"]+'.pth')
        from collections import OrderedDict
        tmp=self.model_save_dir+self.hyper_params["best_model"]+'.pth'
        if tmp.find('checkpoint') >=0:
            state_dict = state_dict['model_state_dict']
            print(' load checkpoint ' )
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.find('module') >= 0:
                name = k[7:] # remove 'module'
            else: name = k
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        self.model.cuda(self.gpuID)
        self.model.eval()  # evaladuation mode
        np.random.seed(seed=self.hyper_params['seed'])  # for reproducibility
        torch.manual_seed(self.hyper_params['seed'])
        torch.cuda.manual_seed(self.hyper_params['seed'])
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.cuda.set_device(self.gpuID)
        with torch.no_grad():
            print('------------------------')
            for module in self.model.children():
                module.train(False)
            #######################################################
            # LOAD IMAGE
            if self.hyper_params['color'] == 1:
                x_original =  cv2.imread(self.iamge_path, 0)
            elif self.hyper_params['color'] == 3:
                x_original = cv2.imread(self.iamge_path, cv2.IMREAD_UNCHANGED)  # BGR or G
            if self.handle_test_size:
                x,x_w_pad_size,x_h_pad_size = utils.pad_to_image(x_original,self.test_mod_size)
            else:
                x = x_original
            x =x.astype(np.float32)/255.0

            y = x + np.random.normal(0, self.hyper_params['sigma']/255.0, x.shape)  # Add Gaussian noise without clipping
            y = y.astype(np.float32)
            if y.ndim == 3:
                y = np.transpose(y, (2,0, 1))
                y_ = torch.from_numpy(y).unsqueeze(0)
            else:
                y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])                            
            y_ = y_.cuda(self.gpuID)
            #########################################################
            # Inference, convert to uint8 and remove pad
            x_ = self.model(y_) 
            x_ = utils.tensor2uint(x_)
            if self.handle_test_size:
                x_ = utils.shave_pad(x_,x_w_pad_size,x_h_pad_size)
            self.no_modulated_image = x_
            if self.hyper_params['color'] == 3:
                self.no_modulated_image = cv2.cvtColor(self.no_modulated_image, cv2.COLOR_BGR2RGB)  # RGB

    #----------------------------------------------------------------------------------------------------------------
    #-----------------------------------Modulation--------------------------------------
    #----------------------------------------------------------------------------------------------------------------
    def ex_test(self,e_0=0,e_1=0,e_2=0,e_3=0,e_4=0):
        np.random.seed(seed=self.hyper_params['seed'])  # for reproducibility
        torch.manual_seed(self.hyper_params['seed'])
        torch.cuda.manual_seed(self.hyper_params['seed'])
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.cuda.set_device(self.gpuID)

        #######################################################
        # Apply lambdas to the GTL
        self.model.layers[0].lambdas=np.array([0,0,0,0,0]).astype(np.float32) # init lambda to zero
        self.model.layers[0].lambdas=np.array([e_0,e_1,e_2,e_3,e_4]).astype(np.float32) # apply user input
        self.model = self.model.cuda(self.gpuID)

        self.model.eval()  # evaladuation mode
        with torch.no_grad():
            print('------------------------')
            for module in self.model.children():
                module.train(False)
            #######################################################
            # LOAD IMAGE
            if self.hyper_params['color'] == 1:
                x_original =  cv2.imread(self.iamge_path, 0)
            elif self.hyper_params['color'] == 3:
                x_original = cv2.imread(self.iamge_path, cv2.IMREAD_UNCHANGED)  # BGR or G
            if self.handle_test_size:
                x,x_w_pad_size,x_h_pad_size = utils.pad_to_image(x_original,self.test_mod_size)
            else:
                x = x_original
            x =x.astype(np.float32)/255.0

            y = x + np.random.normal(0, self.hyper_params['sigma']/255.0, x.shape)  # Add Gaussian noise without clipping
            y = y.astype(np.float32)
            if y.ndim == 3:
                y = np.transpose(y, (2,0, 1))
                y_ = torch.from_numpy(y).unsqueeze(0)
            else:
                y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])                            
            y_ = y_.cuda(self.gpuID)
            #########################################################
            # Inference, convert to uint8 and remove pad
            x_ = self.model(y_) 
            x_ = utils.tensor2uint(x_)
            if self.handle_test_size:
                x_ = utils.shave_pad(x_,x_w_pad_size,x_h_pad_size)
            #######################################################
            # return image
            torch.cuda.empty_cache()

            if self.hyper_params['color'] == 3:
                x_ = cv2.cvtColor(x_, cv2.COLOR_BGR2RGB)  # RGB
                return display(Image.fromarray(x_.astype('uint8'))), display(Image.fromarray(self.no_modulated_image.astype('uint8')))
            return display(Image.fromarray(x_.astype('uint8'), mode='L')),display(Image.fromarray(self.no_modulated_image.astype('uint8'), mode='L'))

