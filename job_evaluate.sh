#!/bin/bash
uname -a
#date
#env
date

DATA_DIRECTORY='/home/sdc/nyu_depth_v2_labeled'
DATA_LIST_PATH='/home/sda/lyx/exp/DORN/dataset/list/nyuv2/test.txt' 
NUM_CLASSES=1 
RESTORE_FROM='./snapshots/LIP_70000.pth'
SAVE_DIR='./outputs_val/' 
INPUT_SIZE='400,300'
GPU_ID=3
 
python evaluate.py --data-dir ${DATA_DIRECTORY} \
                   --data-list ${DATA_LIST_PATH} \
                   --input-size ${INPUT_SIZE} \
                   --is-mirror \
                   --num-classes ${NUM_CLASSES} \
                   --save-dir ${SAVE_DIR} \
                   --gpu ${GPU_ID} \
                   --restore-from ${RESTORE_FROM}
                           
