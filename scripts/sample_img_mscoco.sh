#!/bin/bash


mpiexec -n 2 python sample_img.py  --state 1 \
                --model_path ./models/ \
                --data_path ./mscoco_gen_splits \
                --image_size 64 \
                --guidance_scale 4.0 \
                --batch_size 256 \
                --data_name c-mscoco \
                --noise_schedule "squaredcos_cap_v2" \
                --diffusion_steps 100 \
                --num_classes "6," \