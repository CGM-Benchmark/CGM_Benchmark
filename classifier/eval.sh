#!/bin/bash

CURR_STATE=1
chkpt_path="results_blender/.pt"
GEN_IMAGE_PATH="./_${CURR_STATE}_gen_images_c_blender.npz"
echo $GEN_IMAGE_PATH
python eval.py --dataset c-blender --state $CURR_STATE --checkpoint_path $chkpt_path --im_size 64 \
            --filter_dim 64  --npy_path $GEN_IMAGE_PATH \
            --mode generation


# FOR C-MSCOCO
# CURR_STATE=1
# chkpt_path="results_mscoco/.pt"
# GEN_IMAGE_PATH="./_${CURR_STATE}_gen_images_c_mscoco.npz"
# echo $GEN_IMAGE_PATH
# python eval.py --dataset c-mscoco --state $CURR_STATE --checkpoint_path $chkpt_path --im_size 64 \
#             --filter_dim 64  --npy_path $GEN_IMAGE_PATH \
#             --mode generation

