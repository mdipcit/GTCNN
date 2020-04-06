# Gray 50 GTCNND1
python main.py confs/GTCNN experiment.color=1 experiment.test_only=True experiment.best_model=Gray50_GTCNN_D1 dataset.test_set=[Set12,BSD68,Urban100]
# Gray 50 GTCNND3
python main.py confs/GTCNN experiment.color=1 model.depth=3 experiment.test_only=True experiment.best_model=Gray50_GTCNN_D3 dataset.test_set=[Set12,BSD68,Urban100]
# Gray 50 GTCNND6
python main.py confs/GTCNN experiment.color=1 model.depth=6 model.GTL_stage_option=outconv_slim experiment.test_only=True experiment.best_model=Gray50_GTCNN_D6 dataset.test_set=[Set12,BSD68,Urban100]



# GTCNN-D1
# Gray 30
python main.py confs/GTCNN experiment.sigma=30 experiment.color=1 experiment.test_only=True experiment.best_model=Gray30_GTCNN_D1 dataset.test_set=[Set12,BSD68,Urban100]
# Gray 70
python main.py confs/GTCNN experiment.sigma=70 experiment.color=1 experiment.test_only=True experiment.best_model=Gray70_GTCNN_D1 dataset.test_set=[Set12,BSD68,Urban100]

# Color 50 
python main.py confs/GTCNN experiment.test_only=True experiment.best_model=Color50_GTCNN_D1 

# Color 30
python main.py confs/GTCNN experiment.sigma=30 experiment.test_only=True experiment.best_model=Color30_GTCNN_D1 
# Color 70
python main.py confs/GTCNN experiment.sigma=70 experiment.test_only=True experiment.best_model=Color70_GTCNN_D1
