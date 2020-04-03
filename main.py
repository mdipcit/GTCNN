
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from omegaconf import OmegaConf
import omegaconf
import sys
from train_imai import CNN_train 
from test import Cnn_test 
from datetime import datetime
import os
import numpy as np
def commit(comment):
        
    ###################################
    # Git Comment and Commit
    ###################################
    now = datetime.now()
    date = datetime.now().strftime("%Y/%m/%d-%H:%M:%S")
    comment = comment+"::"+date
    print(comment)
    ###################################
    # Git Push Start
    ###################################
    os.system('git add .')
    os.system('git commit -m '+"'"+comment+"'")
    os.system('git push origin master')
    ###################################
    # Git Commit End
    ###################################
    import subprocess
    cmd = "git rev-parse HEAD"
    hash = subprocess.check_output(cmd.split()).strip().decode('utf-8')
    print(hash)
    return hash

def train_test(EX_conf):
    ###################################
    # Settings 
    ###################################
    # hash_id =commit(EX_conf.experiment.comment) for our experiments
    hash_id = None
    Num_GPUs= len(EX_conf.experiment.device_ids)
    if Num_GPUs>1:
        print("You are using {0} gpu.".format(Num_GPUs))
        print("Automatically multipy lr and batchsize by num of GPUs")
        EX_conf.experiment.lr= EX_conf.experiment.lr*Num_GPUs
        EX_conf.experiment.batchsize= EX_conf.experiment.batchsize*Num_GPUs
        print('Set batchsize :'+str(EX_conf.experiment.batchsize))
        print('Set lr :'+str(EX_conf.experiment.lr))
    # setting experiment
    EX_conf.experiment.git_hash_id = hash_id
    color = '_color'
    if EX_conf.experiment.color==1:
        color = '_gray'
    EX_conf.experiment.name  = EX_conf.model.name+color+'_simga_'+str(EX_conf.experiment.sigma)
    print(EX_conf.experiment.name)
    #----------------------------------------------------------------------------------------------------------------
    OmegaConf.set_readonly(EX_conf, True)
    if not EX_conf.experiment.test_only:
        ###################################
        # Call Train 
        ###################################
        cnn = CNN_train(hyper_params=EX_conf)
        best_model = cnn()
        with omegaconf.read_write(EX_conf):
            EX_conf.experiment.best_model=best_model
    #----------------------------------------------------------------------------------------------------------------

    ###################################
    # Call Test 
    ###################################
    with omegaconf.read_write(EX_conf):
        EX_conf.experiment.name  ="test_"+ EX_conf.experiment.name 
        EX_conf.experiment.seed =0 # test seed

    cnn_test = Cnn_test(EX_conf)
    cnn_test()
    #----------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    print('You must set args like "main.py YamlFileName ConfKey=1 ..."')
    print('------------')
    print('input args')
    print(sys.argv)
    print('------------')
    yaml = sys.argv.pop(1)
    conf = OmegaConf.load(yaml+'.yaml')
    OmegaConf.set_struct(conf, True)
    conf.merge_with_cli()
    # print('Config')
    # print('------------')
    # print(conf.pretty())


    print('------------')
    train_test(conf)