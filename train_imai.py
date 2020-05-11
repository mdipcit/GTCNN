#!/usr/bin/env python
# -*- coding: utf-8 -*-
from comet_ml import Experiment  # We use comet
#  ----------------Import--------------------------
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision.models.utils import load_state_dict_from_url
from torch.utils.data.dataset import Subset,ConcatDataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
import utils
import os
from tqdm import tqdm
import cv2
#  ----------------Import Data Loader--------------------------

import data_generator as dg
from data_generator import DenoisingDataset

#  ----------------Import Models--------------------------
from models import *

#  ----------------Import End--------------------------
try:
    from apex.parallel import DistributedDataParallel 
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    print("If you use fp16/32mix, please install apex from https://www.github.com/nvidia/apex.")

class CNN_train():
    def __init__(self,hyper_params=[]):
        #  ----------------load options--------------------------
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

        # dataset setings
        self.imgSize =self.hyper_params.imgSize
        self.batchsize =self.hyper_params.batchsize
        self.Nsat = self.hyper_params.num_of_img_for_train 
        self.sigma =self.hyper_params.sigma 
        self.model_save_dir = self.hyper_params.model_save_dir 
        self.test_root=self.Data_conf['test_root']
        self.val_root=self.Data_conf['val_root']
        self.train_root=self.Data_conf['train_root']
        self.val_set =  self.Data_conf['val_set']
        self.test_set =  self.Data_conf['test_set']
        self.train_set =  self.Data_conf['train_set']
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)


        #  ----------------load dataset--------------------------
        print(self.val_set)
        self.val_dataset = dg.Get_test_set(self.val_set,self.val_root,self.hyper_params)
        xs = dg.nsat_datagenerator(root=self.train_root,classes = self.train_set, batch_size=self.batchsize, 
        patch_size=self.imgSize,Nsat=self.Nsat,patch_crop=self.hyper_params['patch_crop'],large_size=self.hyper_params['large_size'],stride=self.hyper_params['stride'],scales=self.hyper_params['scales'],color=self.hyper_params['color'])
        self.num_work = 1
        xs = xs.astype('float32')/255.0
        xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))  # tensor of the clean patches, NXCXHXW
        DDataset = DenoisingDataset(xs, self.sigma,self.imgSize,self.hyper_params['random_corp'])
        print('-------------------')
        print('      TRAIN SET    ')
        print(self.train_set)
        print('-------------------')
        print('-------------------')
        print('      Train Data Shapes    ')

        print("len(DDataset)")
        print(len(DDataset))
        print(len(xs))
        print('-------------------')
        print('-------------------')
        torch.manual_seed(self.hyper_params['seed'])
        self.dataloader  = DataLoader(dataset=DDataset, num_workers=self.num_work, collate_fn=DDataset.collate_fn,drop_last=True, batch_size=self.batchsize, pin_memory=True,shuffle=True)


    def __call__(self):
        #######################################
        #           Settings                  #
        #######################################
        gpuID = self.hyper_params['device_ids'][0]
        torch.cuda.set_device(gpuID)
        checkpoint_resume = self.hyper_params['checkpoint_resume']
        checkpoint_PATH = self.hyper_params['checkpoint_PATH']
        opt_level = self.hyper_params['opt_level']
        epoch_num = self.hyper_params['epoch_num']
        device_ids = self.hyper_params['device_ids']
        trained_model = self.hyper_params['best_model']
        lr = self.hyper_params['lr']
        scheduler = self.hyper_params['scheduler']
        if scheduler=='MultiStepLR':
            scheduler_milestones = self.hyper_params['scheduler_milestones']
            scheduler_gamma = self.hyper_params['scheduler_gamma']
        save_checkpoint = self.hyper_params['save_checkpoint']
        handle_test_size = self.hyper_params['handle_test_size'] # pad to test images
        test_mod_size = self.hyper_params['test_mod_size'] # pad to test images
        torch.cuda.empty_cache()
        print('epoch_num:', epoch_num)

        best_psnr=0
        best_model = None
        # model
        np.random.seed(seed=self.hyper_params['seed'])  # for reproducibility
        torch.manual_seed(self.hyper_params['seed'])
        torch.cuda.manual_seed(self.hyper_params['seed'])
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        #################################

        #----------------------------------------------------------------------------------------------------------------
        #-----------------------------------setup model--------------------------------------
        #----------------------------------------------------------------------------------------------------------------
        model = eval(self.Model_conf.name)(self.Model_conf)
        print(self.Model_conf.name)
        criterion = nn.MSELoss()  
        # if self.mode.find('mae') >= 0:
        #     criterion = nn.L1Loss()  
        # if self.mode.find('sum_reduction') >= 0:
        #     criterion = nn.MSELoss(reduction='sum')  

        criterion = criterion.cuda(gpuID)
        # Pallarell mode
        if len(device_ids) > 1:
            optim_params = []
            print(device_ids)
            # device_ids =device_ids
            print("parallel")

            for k, v in model.named_parameters():
                print(k)
            model = model.cuda(gpuID)
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
            if opt_level=="O1" or opt_level=="O2":
                model, optimizer = amp.initialize(model, optimizer,
                                                    opt_level=opt_level
                                                    )
            model = torch.nn.DataParallel(model, device_ids=device_ids)


        # One GPU
        if len(device_ids) < 2:
            optim_params = []
            print(device_ids)
            print("not parallel")
            for k, v in model.named_parameters():
                print(k)
            model = model.cuda(gpuID)
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
            torch.cuda.set_device(gpuID)
            if opt_level=="O1" or opt_level=="O2":
                model, optimizer = amp.initialize(model, optimizer,
                                                    opt_level=opt_level
                                                    )


        #----------------------------- Set  Scheduler------------------------------------------

        if scheduler=='CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)
        elif scheduler=='MultiStepLR':
            # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_milestones, gamma=self.hyper_params['scheduler_gamma'])
        elif scheduler=='ExponentialLR':
            # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hyper_params['scheduler_gamma'])
        print('Param:', utils.count_parameters_in_MB(model))


        # #----------------------------- Set  Val------------------------------------------

        val_interval = self.hyper_params.val_interval
        # if epoch_num>600:
        #     val_interval =10
        # if epoch_num>2000:
        #     val_interval =50

        #----------------------------------------------------------------------------------------------------------------
        #-----------------------------------experiment.train--------------------------------------
        #----------------------------------------------------------------------------------------------------------------
        with self.experiment.train():
            step_iter=0
            start_epoch=1
            # checkpoint
            if checkpoint_resume==True:
                tmp=self.model_save_dir+checkpoint_PATH+'.pth'
                checkpoint = torch.load(tmp)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint.keys():
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if opt_level=="O1" or opt_level=="O2":
                    if 'scheduler_state_dict' in checkpoint.keys():
                        amp.load_state_dict(checkpoint['amp'])
                    else:
                        print('this state_dict dose not have amp')
                start_epoch = checkpoint['epoch']
                print('resume checkpoint from:'+str(start_epoch))

            # Train loop
            for epoch in range(start_epoch, epoch_num+1):
                start_time = time.time()
                print('epoch', epoch)
                step_iter=0
                train_loss = 0
                mse_train_loss = 0
                for module in model.children():
                    module.train(True)

                print('----------------------')

                for ite, (input, target) in enumerate(tqdm(self.dataloader)):
                    lr_patch = input.cuda(gpuID)
                    hr_patch = target.cuda(gpuID)
                    optimizer.zero_grad()
                    output = model(lr_patch)
                    mse_loss = criterion(output, hr_patch)
                    loss=mse_loss
                    loss_item=loss.item()
                    self.experiment.log_metric("Train Loss", loss_item, step=step_iter)   
                    if opt_level=="O1" or opt_level=="O2":
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            self.experiment.log_metric("amp.scale_loss ", scaled_loss.item(), step=step_iter)   
                            scaled_loss.backward()

                    else:
                        loss.backward()
                    optimizer.step()
                    train_loss += loss_item
                    mse_train_loss += mse_loss.item()
                    step_iter+=1
                    
                self.experiment.log_metric("Averageloss", train_loss/step_iter, step=epoch)
                self.experiment.log_metric("Sum  loss", mse_train_loss, step=epoch)
                print('Train set : Average loss: {:.4f}'.format(train_loss))
                print('time ', time.time()-start_time)
                # update lr after update optime 
                scheduler.step(epoch)

                #----------------------------------------------------------------------------------------------------------------
                #-----------------------------------experiment.validate--------------------------------------
                #----------------------------------------------------------------------------------------------------------------
                with self.experiment.validate():
                    if epoch % val_interval == 0 : 
                        model.eval()  # evaluation mode
                        with torch.no_grad():
                            print('------------------------')
                            for module in model.children():
                                module.train(False)

                            test_ite = 0
                            test_psnr = 0
                            test_ssim = 0
                            psnrs_all = []
                            for label_index,val_data in enumerate(self.val_dataset):
                                set_cur = val_data[0]
                                file_list = val_data[1]
                                print(set_cur)
                                psnrs = []
                                ssims = []
                                im_num=0
                                for im in file_list:
                                    x_original, y_, x_w_pad_size, x_h_pad_size = im

                                    torch.cuda.synchronize()
                                    start_time = time.time()

                                    # -------------------------inference ----------------

                                    y_ = y_.cuda(gpuID)
                                    x_ = model(y_)  # inference
                                    x_ = utils.tensor2uint(x_)

                                    torch.cuda.synchronize()
                                    elapsed_time = time.time() - start_time

                                    # -------------------------calc ----------------

                                    if handle_test_size:
                                        x_ = utils.shave_pad(x_,x_w_pad_size,x_h_pad_size)
                                    psnr_x_ = utils.calculate_psnr( x_,x_original)
                                    ssim_x_ = utils.calculate_ssim( x_,x_original)

                                    psnrs.append(psnr_x_)
                                    psnrs_all.append(psnr_x_)
                                    ssims.append(ssim_x_)
                                    # self.experiment.log_metric("per_psnr_im_num"+str(im_num), psnr_x_, step=epoch)

                                psnr_avg = np.mean(psnrs)
                                ssim_avg = np.mean(ssims)
                                self.experiment.log_metric("VAL_PSNR_"+set_cur, psnr_avg, step=epoch)

                            psnrs_all_avg = np.mean(psnrs_all)
                            self.experiment.log_metric("psnr_avg_entir_val_sets", psnrs_all_avg, step=epoch)

                            if psnrs_all_avg>best_psnr:
                                self.experiment.log_metric("best_psnr", psnrs_all_avg, step=epoch)
                                # best_psnr=psnr_avg
                                best_psnr=psnrs_all_avg
                                print('------------------------')
                                print('-------best_psnr---------')
                                print(best_psnr)
                                best_model='best_%s_%d' % (self.hyper_params.name, int(epoch))
                            if save_checkpoint: 
                                save_model_name='checkpoint_%s_%d' % (self.hyper_params.name, int(epoch))
                                if opt_level=="O1" or opt_level=="O2":
                                    torch.save({ 'epoch': epoch+1,
                                                'model_state_dict': model.state_dict(),
                                                'optimizer_state_dict': optimizer.state_dict(),
                                                'scheduler_state_dict': scheduler.state_dict(),
                                                'amp': amp.state_dict()
                                                }, self.model_save_dir+save_model_name+'.pth')
                                else:
                                    torch.save({ 'epoch': epoch+1,
                                                'model_state_dict': model.state_dict(),
                                                'optimizer_state_dict': optimizer.state_dict(),
                                                'scheduler_state_dict': scheduler.state_dict(),
                                                }, self.model_save_dir+save_model_name+'.pth')

#----------------------------------------------------------------------------------------------------------------
#-----------------------------------end--------------------------------------
#----------------------------------------------------------------------------------------------------------------
        torch.save(model.state_dict(),self.model_save_dir+best_model+'.pth')
        torch.cuda.empty_cache()
        return best_model

