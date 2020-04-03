#!/usr/bin/env python
# -*- coding: utf-8 -*-
from comet_ml import Experiment

import os
import csv
import logging
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import seaborn as sns
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torch.autograd import Variable

import os
import utils
import logging
import os, time, datetime
import torch.nn.functional as F
from os import listdir
import glob
import sys
import cv2

#  ----------------Import Models--------------------------
from models import *
try:
    from apex.parallel import DistributedDataParallel 
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    print("If you use fp16/32mix, please install apex from https://www.github.com/nvidia/apex.")

#----------------------------------------------------------------------------------------------------------------
#-----------------------------------main process--------------------------------------
#----------------------------------------------------------------------------------------------------------------

class Cnn_test():
    def __init__(self,hyper_params=[]):
        self.Experiment_Confs = hyper_params
        self.Model_conf =  hyper_params.model
        self.Data_conf =  hyper_params.dataset
        self.hyper_params = hyper_params.experiment

        # comet ml
        self.experiment = Experiment(api_key="",
                                project_name=self.hyper_params.comet_project_name, workspace="",disabled=self.hyper_params.comet_disabled)
        self.experiment.log_parameters(self.hyper_params)
        self.experiment.set_name(self.hyper_params.name)
        self.experiment.log_parameters(self.Data_conf)
        self.experiment.log_parameters(self.Model_conf)

        self.model_save_dir = self.hyper_params['model_save_dir']
        self.gpuID=self.hyper_params['device_ids'][0]
        self.test_root=self.Data_conf['test_root']
        self.test_set=self.Data_conf['test_set']
        self.saveImage=self.hyper_params["saveImage"]

    #----------------------------------------------------------------------------------------------------------------
    #-----------------------------------experiment.test--------------------------------------
    #----------------------------------------------------------------------------------------------------------------
    def ex_test(self,model_file_name,test_sets):
        handle_test_size =  self.hyper_params['handle_test_size']
        test_mod_size = self.hyper_params['test_mod_size']
        with self.experiment.test():
            #-----------------------------------setup model--------------------------------------
            model = eval(self.Model_conf.name)(self.Model_conf)
            print(model_file_name)
            # for k, v in model.named_parameters():
            #     print(k)
            # print('Param:', utils.count_parameters_in_MB(model))

            state_dict = torch.load(self.model_save_dir+model_file_name+'.pth')
            from collections import OrderedDict
            tmp=self.model_save_dir+model_file_name+'.pth'
            if tmp.find('checkpoint') >=0:
                state_dict = state_dict['model_state_dict']
                print(' load checkpoint ' )
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.find('module') >= 0:
                    name = k[7:] # remove 'module'
                else: name = k
                new_state_dict[name] = v
                # print(name)
            # print(model)
            model.load_state_dict(new_state_dict)


            model = model.cuda(self.gpuID)

            model.eval()  # evaladuation mode
            with torch.no_grad():
                print('------------------------')
                for module in model.children():
                    module.train(False)
                test_ite = 0
                test_psnr = 0
                test_ssim = 0
                psnrs = []
                ssims = []
                class_PNSRS= []
                all_PSNR= []
                class_SSIMS= []
                for label_index,set_cur in enumerate(test_sets):
                    print(set_cur)
                    psnrs = []
                    ssims = []
                    im_num=0
                    file_list = sorted(os.listdir(os.path.join(self.test_root,set_cur)))
                    
                    for im in file_list:
                        if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
                            im_num+=1
                            if self.hyper_params['color'] == 1:
                                x_original =  cv2.imread(os.path.join(self.test_root ,set_cur, im), 0)
                            elif self.hyper_params['color'] == 3:
                                x_original = cv2.imread(os.path.join(self.test_root ,set_cur, im), cv2.IMREAD_UNCHANGED)  # BGR or G

                            if handle_test_size:
                                x,x_w_pad_size,x_h_pad_size = utils.pad_to_image(x_original,test_mod_size)
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
                            torch.cuda.synchronize()
                            start_time = time.time()
                            y_ = y_.cuda(self.gpuID)
                            x_ = model(y_)  # inference
                            x_ = utils.tensor2uint(x_)
                            

                            torch.cuda.synchronize()
                            elapsed_time = time.time() - start_time
                            # print('%10s : %10s : %2.4f second' % ( set_cur, im, elapsed_time))

                            if handle_test_size:
                                x_ = utils.shave_pad(x_,x_w_pad_size,x_h_pad_size)
                            psnr_x_ = utils.calculate_psnr( x_,x_original)
                            ssim_x_ = utils.calculate_ssim( x_,x_original)
                            if self.saveImage:
                                if self.hyper_params['color'] == 3:
                                    x_ = cv2.cvtColor(x_, cv2.COLOR_BGR2RGB)  # RGB
                                if not os.path.exists('./savedImages/'+model_file_name+'/+set_cur'):
                                    os.makedirs('./savedImages/'+ model_file_name+'/+set_cur')
                                utils.imsave(x_,'./savedImages/'+ model_file_name+'/+set_cur'+'/'+im)


                            all_PSNR.append(psnr_x_)
                            psnrs.append(psnr_x_)
                            ssims.append(ssim_x_)
                    psnr_avg = np.mean(psnrs)
                    ssim_avg = np.mean(ssims)
                    print("test_psnr_"+set_cur, psnr_avg)
                    print("test_ssim_"+set_cur, ssim_avg)
                    self.experiment.log_metric("test_psnr_"+set_cur, psnr_avg)
                    self.experiment.log_metric("test_ssim_"+set_cur, ssim_avg)
            self.experiment.log_metric("entir_test_psnr", np.array(all_PSNR))
            self.experiment.log_metric("entir_test_avg_psnr", np.mean(all_PSNR))

    def __call__(self):
        np.random.seed(seed=self.hyper_params['seed'])  # for reproducibility
        torch.manual_seed(self.hyper_params['seed'])
        torch.cuda.manual_seed(self.hyper_params['seed'])
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        self.ex_test(self.hyper_params["best_model"],self.test_set)
