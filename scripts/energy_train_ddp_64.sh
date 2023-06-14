#!/bin/bash



python -m torch.distributed.launch --nproc_per_node=2 --master_port=5672  --use_env train_ddp.py \
  --project_name 'benchmark-cgm' \
  --energy_mode \
  --batch_size 32  \
  --diffusion_steps 100 \
  --cl_states 3 \
  --noise_schedule 'linear' \
  --seed 42 \
  --test_guidance_scale 8 \
  --epochs 50 \
  --log_frequency 50 \
  --num_classes "3," \
  --learning_rate 1e-04 \
  --side_x 64 \
  --side_y 64 \
  --resize_ratio 1.0 \
  --uncond_p 0.05 \
  --checkpoints_dir './RevDiff_May_NODup_blender_3states_64_Linear_100step_full_replay' \
  --outputs_dir './RevDiff_May_NODup_blender_3states_64_Linear_100step_full_replay'


