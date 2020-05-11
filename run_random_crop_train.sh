# Gray 50 GTCNND1
python main.py confs/GTCNN experiment.color=1 experiment.random_corp=True experiment.large_size=512 experiment.stride=512 dataset.test_set=[Set12,BSD68,Urban100] experiment.epoch_num=2500
# Gray 50 GTCNND3
python main.py confs/GTCNN experiment.color=1 experiment.random_corp=True experiment.large_size=512 experiment.stride=512   model.depth=3 experiment.batchsize=30 dataset.test_set=[Set12,BSD68,Urban100] experiment.epoch_num=2500
# Gray 50 GTCNND6
python main.py confs/GTCNN experiment.color=1 experiment.random_corp=True experiment.large_size=512 experiment.stride=512   model.depth=6 experiment.batchsize=12 model.GTL_stage_option=outconv_slim  dataset.test_set=[Set12,BSD68,Urban100] experiment.epoch_num=2500

###########
# GTCNN-D1
###########
# Gray 30
python main.py confs/GTCNN experiment.sigma=30 experiment.random_corp=True experiment.large_size=512 experiment.stride=512   experiment.color=1  dataset.test_set=[Set12,BSD68,Urban100] experiment.epoch_num=2500
# Gray 70
python main.py confs/GTCNN experiment.sigma=70 experiment.random_corp=True experiment.large_size=512 experiment.stride=512   experiment.color=1  dataset.test_set=[Set12,BSD68,Urban100] experiment.epoch_num=2500

# Color 50 
python main.py confs/GTCNN experiment.random_corp=True experiment.large_size=512 experiment.stride=512 experiment.epoch_num=2500     
# Color 30
python main.py confs/GTCNN experiment.random_corp=True experiment.large_size=512 experiment.stride=512 experiment.sigma=30 experiment.epoch_num=2500   
# Color 70
python main.py confs/GTCNN experiment.sigma=70   experiment.random_corp=True experiment.large_size=512 experiment.stride=512 experiment.epoch_num=2500  
