Code can be downloaded from: 
https://www.yunpan.com/surl_ytiysWmjQJR （Code：fd4a, Password: The name of our paper）

Inference: 
python data/ShanghaiTech/pose/multi_cls_auc.py 

Directory including trajectories and sequences of poses:

Training:
python train_statespace.py --smooth_or_predwithsmoothed_or_predwithunsmoothed train 

Inference for human class:
python train_statespace.py --smooth_or_predwithsmoothed_or_predwithunsmoothed predwithunsmoothed 

Revise sequence length:
revise input_frame in train_statespace.py 

Implementation of combining postures with displacement: 
def normalize_data_transformed  in  models/training_state.py 
