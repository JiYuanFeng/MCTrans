#!/usr/bin/env bash
CFG="configs/debug/unet_vgg32_d5_256x256_400ep_pannuke.py"
WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
python3 tools/train.py ${CFG} --work-dir ${WORK_DIR} --gpus 1

