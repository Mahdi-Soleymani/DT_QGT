#!/bin/bash
# Activate Conda environment
source /opt/anaconda3/bin/activate npss  # Use `conda activate npss` if needed

for seed in 123 231 312
do
    /opt/anaconda3/envs/npss/bin/python run_QGT.py   --seed $seed  --epochs 2 --learning_rate 5e-5 --batch_size 64

done

conda deactivate

