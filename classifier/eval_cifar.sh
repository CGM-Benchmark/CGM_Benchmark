#!/bin/bash

chkpt_path="results_cifar10/.pt"
GEN_IMAGE_FOLDER="./cifar10_gens"
python eval_cifar.py --chkpt_path $chkpt_path --gen_imgs_folder $GEN_IMAGE_FOLDER \
                     --splits 10