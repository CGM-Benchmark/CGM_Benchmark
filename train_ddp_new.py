#  Description: This file is used to train the model in a distributed manner.

import argparse
from glob import glob
import os

import numpy as np
import torch as th
from tqdm import trange
import random
from cl_utils.finetune_ddp import run_cl_epoch
from cl_utils.util import load_model
from cl_utils.loader import blender_cl_64,COCO_64_cl,domain_cl
from cl_utils.train_util import wandb_setup


import torch 
import utils

def run_experiment(
    diffusion_steps,
    world_size,
    dist_url,
    learn_sigma: bool,
    uncond = False,
    noise_schedule="squaredcos_cap_v2",
    data_dir="./data",
    batch_size=1,
    learning_rate=1e-5,
    resume_ckpt="",
    checkpoints_dir="./finetune_checkpoints",
    device="cpu",
    project_name="cgm_blender",
    num_epochs=100,
    log_frequency=100,
    sample_bs=1,
    sample_gs=8.0,
    outputs_dir = "./outputs",
    num_states= 3,
    num_classes="",
    energy_mode=False, 
    buffer=True,
    gradient_accumualation_steps =1,
    clip_grad_norm = False,
    data_name = "",
    uncond_p = 0.05,
    is_ae = True,
):

    is_master = (utils.get_rank()==0)
    device = torch.device("cuda")


    # Start wandb logging
    if is_master:
        wandb_run = wandb_setup(
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            base_dir=checkpoints_dir,
            project_name=project_name,
            learn_sigma = learn_sigma
        )
        print("Wandb setup.")
    else:
        wandb_run = None

   
    # Model setup
    model, diffusion, model_options = load_model(
        diffusion_steps=diffusion_steps,
        energy_mode=energy_mode,
        noise_schedule=noise_schedule,
        learn_sigma=learn_sigma,
        num_classes=num_classes,
        resume=resume_ckpt,
        model_type="base",
        data_name=data_name,
        is_ae = is_ae,
    )

    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    if is_master == 0:
        number_of_params = sum(x.numel() for x in model.parameters())
        print(f"Number of parameters: {number_of_params}")
        number_of_trainable_params = sum(
            x.numel() for x in model.parameters() if x.requires_grad
        )
        print(f"Trainable parameters: {number_of_trainable_params}")
        
        # Watch the model for 0 rank 

        # wandb_run.watch(model, log="all")
    


    optimizer = th.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training setup

    if is_master == 0:
        print("buffer status: ", buffer)
    

    for state in range(1,num_states+1):
        if is_master:
            print("Current State: ", state)
        
        # Get the dataset on current task
        if data_name=="c-blender":
            dataset = blender_cl_64(
                    data_path=data_dir,
                    buffer = True,
                    state=state,
                    uncond_p = uncond_p,
                )        
        elif data_name=="c-MSCOCO":
            dataset = COCO_64_cl(
                data_path= data_dir,
                buffer = True,
                state=state,
                uncond_p = uncond_p,
            )
        elif data_name=="c-domain":
            dataset = domain_cl(
                data_path= data_dir,
                buffer = True,
                state=state,
                uncond_p = uncond_p,
            )
        else:
            raise ValueError("Unknown dataset: ", data_name)

    
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()

        sampler_train = torch.utils.data.DistributedSampler(
                dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
    
   
        dataloader = torch.utils.data.DataLoader(
            dataset, sampler=sampler_train,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
                        )

        
        if data_name=="c-blender":
            test_prompt_lab= dataset.get_test_sample(state)
            class_id_inf = state-1
        elif data_name == "c-MSCOCO":
            test_prompt_lab,class_id_inf= dataset.get_test_sample(state)
        elif data_name == "c-domain":
            test_prompt_lab,class_id_inf= dataset.get_test_sample(state)

        for epoch in trange(num_epochs):
            if is_master:
                print(f"Starting epoch {epoch}")

            dataloader.sampler.set_epoch(epoch)
            run_cl_epoch(
                is_master = is_master,
                data_name = data_name,
                class_id=class_id_inf,
                uncond = uncond,
                state= state,
                device = device,
                model=model,
                diffusion=diffusion,
                options=model_options,
                optimizer=optimizer,
                dataloader=dataloader,
                prompt=test_prompt_lab,
                sample_bs=sample_bs,
                sample_gs=sample_gs,
                checkpoints_dir=checkpoints_dir,
                outputs_dir=outputs_dir,
                wandb_run=wandb_run,
                log_frequency=log_frequency,
                epoch=epoch,
                gradient_accumualation_steps=gradient_accumualation_steps,
                energy_mode= energy_mode,
                clip_grad_norm = clip_grad_norm,
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-data", type=str, default="./data")
    parser.add_argument("--data_name", type=str, default=None)
    parser.add_argument("--batch_size", "-bs", type=int, default=1)
    parser.add_argument("--learning_rate", "-lr", type=float, default=2e-5)
    parser.add_argument(
        "--uncond_p",
        "-p",
        type=float,
        default=0.05,
        help="Probability of using the empty/unconditional token instead of a caption. OpenAI used 0.2 for their finetune.",
    )
    parser.add_argument(
        "--resume_ckpt",
        "-resume",
        type=str,
        default="",
        help="Checkpoint to resume from",
    )
    parser.add_argument(
        "--checkpoints_dir", "-ckpt", type=str, default="./glide_checkpoints/"
    )
    parser.add_argument(
        "--outputs_dir",  type=str, default="./glide_outs/"
    )
    parser.add_argument(
        "--num_classes",  type=str, default=""
    )
    parser.add_argument("--device", "-dev", type=str, default="")
    parser.add_argument("--log_frequency", "-freq", type=int, default=100)
    parser.add_argument("--project_name", "-name", type=str, default="cgm-blender")
    parser.add_argument("--use_captions", "-txt", action="store_true")
    parser.add_argument("--learn_sigma", action="store_true")
    parser.add_argument("--uncond", action="store_true")
    parser.add_argument("--buffer", action="store_true")
    parser.add_argument("--epochs", "-epochs", type=int, default=60)
    parser.add_argument(
        "--test_prompt",
        "-prompt",
        type=str,
        default="a group of skiers are preparing to ski down a mountain.",
    )
    parser.add_argument(
        "--test_batch_size",
        "-tbs",
        type=int,
        default=1,
        help="Batch size used for model eval, not training.",
    )
    parser.add_argument(
        "--test_guidance_scale",
        "-tgs",
        type=float,
        default=4.0,
        help="Guidance scale used during model eval, not training.",
    )

    
    parser.add_argument(
        "--energy_mode",
        action="store_true",
        help="Energy_mode",
    )
    
    
    parser.add_argument("--seed", "-seed", type=int, default=0)
    parser.add_argument(
        "--cudnn_benchmark",
        "-cudnn",
        action="store_true",
        help="Enable cudnn benchmarking. May improve performance. (may not)",
    )

    parser.add_argument(
        "--cl_states",type=int, default=3, help="number of states"
    )
    parser.add_argument(
        "--buffer_size",type=int, default=1000, help="Buffer(Replay) Size"
    )
    parser.add_argument(
        "--noise_schedule",  type=str, default="squaredcos_cap_v2",choices=["squaredcos_cap_v2","linear"]
    )
   
    parser.add_argument(
        "--world_size",type=int, default=3, help="number of states"
    )
    parser.add_argument(
        "--dist_url",  type=str, default="env://"
    )
    parser.add_argument(
        "--diffusion_steps",type=int, default=0, help="diffusion_steps"
    )
    parser.add_argument(
        "--gradient_accumualation_steps",type=int, default=1, help="gradient_accumulation_steps"
    )
    
    parser.add_argument(
        "--clip_grad_norm",
        action="store_true",
        help="Clipping the Gradient to 10",
    )
    parser.add_argument(
        "--is_ae",
        action="store_true",
        help="is_ae",
    )
    args = parser.parse_args()

    return args

def setup_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # th.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # CUDA/CPU setup
    args = parse_args()

    utils.init_distributed_mode(args)
 

    seeds= args.seed + utils.get_rank() + 10
    print("Setting the Seed to ", seeds)
    setup_seed(seed=seeds)

    for arg in vars(args):
        print(f"--{arg} {getattr(args, arg)}")


    data_dir = args.data_dir
        
    run_experiment(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        uncond_p=args.uncond_p,
        resume_ckpt=args.resume_ckpt,
        checkpoints_dir=args.checkpoints_dir,
        log_frequency=args.log_frequency,
        project_name=args.project_name,
        num_epochs=args.epochs,
        sample_bs=args.test_batch_size,
        sample_gs=args.test_guidance_scale,
        outputs_dir =args.outputs_dir,
        data_name = args.data_name,
        num_states=args.cl_states,
        num_classes = args.num_classes,
        buffer = args.buffer,
        learn_sigma = args.learn_sigma,
        noise_schedule = args.noise_schedule,
        uncond = args.uncond,
        energy_mode = args.energy_mode,
        world_size = args.world_size,
        dist_url = args.dist_url,
        diffusion_steps=args.diffusion_steps,
        gradient_accumualation_steps=args.gradient_accumualation_steps,
        clip_grad_norm = args.clip_grad_norm,
        is_ae = args.is_ae,
    )
