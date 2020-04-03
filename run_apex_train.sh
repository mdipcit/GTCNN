# Gray 50 GTCNND1
python main.py confs/GTCNN experiment.color=1 experiment.opt_level=O1 dataset.test_set=[Set12,Set68,Urban100]
# Gray 50 GTCNND3
python main.py confs/GTCNN experiment.color=1 experiment.opt_level=O1 model.depth=3 experiment.batchsize=30 dataset.test_set=[Set12,Set68,Urban100]
# Gray 50 GTCNND6
python main.py confs/GTCNN experiment.color=1 experiment.opt_level=O1 model.depth=6 experiment.batchsize=12 model.GTL_stage_option=outconv_slim  dataset.test_set=[Set12,Set68,Urban100]


###########
# GTCNN-D1
###########
# Gray 30
python main.py confs/GTCNN experiment.sigma=30 experiment.opt_level=O1  experiment.color=1  dataset.test_set=[Set12,Set68,Urban100]
# Gray 70
python main.py confs/GTCNN experiment.sigma=70 experiment.opt_level=O1  experiment.color=1  dataset.test_set=[Set12,Set68,Urban100]

# Color 50 
python main.py confs/GTCNN experiment.opt_level=O1    
# Color 30
python main.py confs/GTCNN experiment.opt_level=O1  experiment.sigma=30   
# Color 70
python main.py confs/GTCNN experiment.opt_level=O1  experiment.sigma=70  
