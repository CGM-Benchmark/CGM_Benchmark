#!/bin/bash


#increment_classes:  1 for 10 task splits, 2 for 5 task splits

python -m torch.distributed.launch --nproc_per_node=2 --master_port=5672  --use_env train_ddp_cifar.py \
  --project_name 'benchmark-cgm' \
  --increment_classes 1 \
  --buffer \
  --batch_size 128  \
  --buffer_size 2000 \
  --herding_method 'random' \
  --diffusion_steps 1000 \
  --noise_schedule 'squaredcos_cap_v2' \
  --seed 42 \
  --test_guidance_scale 4 \
  --epochs 50 \
  --log_frequency 10 \
  --num_classes "10," \
  --learning_rate 1e-04 \
  --checkpoints_dir './c10_CGM_May_cosine_10tasks' \
  --outputs_dir 'c10_CGM_May_cosine_10tasks'