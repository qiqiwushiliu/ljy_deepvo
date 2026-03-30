#!/bin/bash

cd /home/LXT/LJY/DeepVO-pytorch

# echo "=========================================="
# echo "Testing LSTM model..."
# echo "=========================================="
# python test.py --model _lstm

echo ""
echo "=========================================="
echo "Testing CfC model..."
echo "=========================================="
python test.py --model _cfc

echo ""
echo "=========================================="
echo "All tests completed! Results saved in:"
echo "/home/LXT/LJY/DeepVO-pytorch/results/"
echo "=========================================="
ls -la /home/LXT/LJY/DeepVO-pytorch/results/

# cd /home/LXT/LJY && ./run_test.sh
