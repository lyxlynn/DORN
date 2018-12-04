#!/bin/bash
uname -a
#date
#env
#NETWORK='psp'
NETWORK='baseline'
SAVE_DIR='./outputs_val/' 
DATA_DIRECTORY='./home/sdc/nyu_depth_v2_train_fill_all'
DATA_LIST_PATH='/home/sdc/nyu_depth_v2_train_fill_all/train_rgb.txt'
RESTORE_FROM='./dataset/MS_DeepLab_resnet_pretrained_init.pth'
#RESTORE_FROM='./dataset/resnet101-imagenet.pth'
#SNAPSHOT_DIR='./snapshots_train_eval/'
SNAPSHOT_DIR=$NETWORK
LR=1e-2
BATCHSIZE=2
STEPS=593412   # 20 epoch
SAVE_PRED_EVERY=400
GPU_IDS='2'
INPUT_SIZE='400,300'
NUM_CLASSES=1  
STARTITERS=0
 
python tfrecord.py --data-dir ${DATA_DIRECTORY} \
                          --data-list ${DATA_LIST_PATH} \
                          --input-size ${INPUT_SIZE} \
                          --num-classes ${NUM_CLASSES} \
                          --random-mirror \
                          --random-scale \
                          --gpu ${GPU_IDS} \
                          --learning-rate ${LR} \
                          --batch-size ${BATCHSIZE} \
                          --num-steps ${STEPS} \
                          --start-iters ${STARTITERS}\
                          --restore-from ${RESTORE_FROM} \
                          --snapshot-dir ${SNAPSHOT_DIR} \
                          --save-pred-every ${SAVE_PRED_EVERY}\
		          --network ${NETWORK}
