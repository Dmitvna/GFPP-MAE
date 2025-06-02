# GFPP-MAE
The code for paper "GFPP-MAE: Gradient-Guided Frequency Reconstruction and Position Predictions Advance MAE for 3D CT Image Segmentation".

## Preparation
### Dataset
1. Download the BTCV dataset from https://pan.baidu.com/s/1rATxY3q_4nCrbpTl_IFHhw?pwd=8nxn
2. Create a "data" folder in the project folder and move the downloaded dataset into it. The project folder structure is as follows:

              |GFPP-MAE  
              ----|data  
              --------|BTCV  
              ------------|imagesTr  
              ------------|imagesTs  
              ------------|labelsTr  
              ------------|dataset_0.json  
              ----|configs  
              ----|demo   
              ----|lib  
              ----|main.py  
              ----|README.md

### Training weights
1. Download the pre-train weights "checkpoint_9999.pth.tar" from https://pan.baidu.com/s/1I4lEbOYKnmhD2YvUOSXwxg?pwd=x37r
2. Download the fine-tune weights "best_model.pth.tar" from https://pan.baidu.com/s/1x20sTFgXkSvp5Z644bEnFg?pwd=q3fy

## Pre-train in 1 GPU
python main.py configs/gfpp-mae_btcv_1gpu.yaml --run_name=gfpp-mae

## Fine-tune in 1 GPU
python main.py configs/unetr_btcv_1gpu.yaml --run_name=unetr --pretrain=XXX/checkpoint_9999.pth.tar

## Evaluate and visualize
python main.py configs/unetr_btcv_1gpu.yaml --run_name=test --test=True --pretrain=XXX/best_model.pth.tar
   
