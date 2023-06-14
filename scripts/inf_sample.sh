

# # For the energy based parametrization
# python inf_sample.py --ckpt_path  "glide-ft-49x530_3.pt" \
#                         --energy_mode


# For the score based parametrization
# First unzip the c_blender_linear_100step_state3_chkpt.tar.gz file ot get the checkpoint "glide-ft-49x530_3.pt"
python inf_sample.py --ckpt_path   "/glide-ft-49x530_3.pt" \
                       