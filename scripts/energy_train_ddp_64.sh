#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=2 --master_port=5672  --use_env train_ddp_new.py \
  --project_name 'benchmark-cgm' \
  --energy_mode \
  --is_ae \
  --data_name 'c-blender' \
  --gradient_accumualation_steps 1 \
  --batch_size 64  \
  --data_dir "./BLENDER_PREPRINT_GEN_30k_FINAL_DATASET.npz" \
  --diffusion_steps 100 \
  --cl_states 3 \
  --noise_schedule 'linear' \
  --seed 42 \
  --test_guidance_scale 8 \
  --epochs 50 \
  --log_frequency 50 \
  --num_classes "3," \
  --learning_rate 1e-04 \
  --uncond_p 0.05 \
  --checkpoints_dir './logs/c_blender_energy_dae_buffer' \
  --outputs_dir './logs/c_blender_energy_dae_buffer'


