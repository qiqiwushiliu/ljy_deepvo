#!/bin/bash

cd /home/LXT/LJY/DeepVO-pytorch

echo "=========================================="
echo "Training DeepVO with CfC/NCP..."
echo "=========================================="
CUDA_VISIBLE_DEVICES=1 python main_cfc_ncp.py

echo ""
echo "=========================================="
echo "Training DeepVO with LSTM..."
echo "=========================================="
CUDA_VISIBLE_DEVICES=1 python main.py
# cd /home/LXT/LJY && ./run_train.sh