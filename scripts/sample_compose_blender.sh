#!/bin/bash


python sample_compose.py --state 1 \
                        --ckpt_path ./models/ \
                        --diffusion_steps 100 \
                        --noise_schedule "linear" \
                        --data_name 'c-blender' \
                        --num_samples 100 \
                        --sampler 'UHMC' 