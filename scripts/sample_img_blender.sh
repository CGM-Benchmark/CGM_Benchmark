#!/bin/bash

mpiexec -n 2 python sample_img.py  --state 1 \
                --model_path ./models/ \
                --data_path ./Custom_blender_dataset_cl.npz \
                --eval_idx_path ./custom_blender_eval_5k_IDX.npy \
                --image_size 64 \
                --guidance_scale 12.0 \
                --batch_size 256 \
                --data_name c-blender \
                --noise_schedule "squaredcos_cap_v2" \
                --diffusion_steps 100 \
                --num_classes "3," \

