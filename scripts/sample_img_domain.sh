#!/bin/bash


# State 1 to 10 , for covering all classes of C-Domain
mpiexec -n 2 python sample_img_domain.py  --state 1 \
                --model_path ./models/ \
                --image_size 32 \
                --guidance_scale 4.0 \
                --batch_size 256 \
                --data_name c-domain \
                --noise_schedule "squaredcos_cap_v2" \
                --diffusion_steps 1000 \
                --num_classes "10," \

