#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=2 --master_port=5674  --use_env train_ddp.py \
  --project_name 'benchmark-cgm' \
  --data_name 'c-domain' \
  --batch_size 64  \
  --gradient_accumualation_steps 1 \
  --diffusion_steps 1000 \
  --cl_states 3 \
  --noise_schedule 'squaredcos_cap_v2' \
  --seed 42 \
  --test_guidance_scale 4 \
  --epochs 50 \
  --log_frequency 100 \
  --num_classes "10," \
  --learning_rate 1e-04 \
  --checkpoints_dir './custom_domain_score' \
  --outputs_dir './custom_domain_score'
