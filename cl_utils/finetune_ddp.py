import os
from typing import Tuple

import torch as th
from wandb import wandb

from cl_utils import train_util, util
import torch
import torch.distributed as dist

def avg_metric(metric):
    """
    Computes average metric value gathered from workers.
    :param metric: input tensor
    :return: averaged tensor
    """
    lst = [torch.zeros_like(metric) for _ in range(dist.get_world_size())]
    dist.all_gather(lst, metric)
    avg = torch.stack(lst).mean().view_as(metric)
    return avg


def base_train_step(
    device,
    model ,
    diffusion,
    batch: Tuple[th.Tensor, th.Tensor, th.Tensor],
    uncond = False,
    learn_sigma=False,
):
    """
    Perform a single training step.

        Args:
            model: The model to train.
            diffusion: The diffusion to use.
            batch: A tuple of (labels, masks, reals) 
        Returns:
            The loss.
    """
    labels, masks, reals = [x.to(device) for x in batch]
    
   
    timesteps = th.randint(
        0, len(diffusion.betas) - 1, (reals.shape[0],), device=device
    )
    noise = th.randn_like(reals, device=device)
    x_t = diffusion.q_sample(reals, timesteps, noise=noise).to(device)
    _, C = x_t.shape[:2]


    if uncond:
        model_output = model(
            x_t.to(device),
            timesteps.to(device),
            y=None,
            masks=None,
        )
    else:
        model_output = model(
            x_t.to(device),
            timesteps.to(device),
            y=labels.to(device),
            masks=masks.to(device),
        )    

    if not learn_sigma:
        epsilon = model_output    
    else:
        epsilon, _ = th.split(model_output, C, dim=1)
    return th.nn.functional.mse_loss(epsilon, noise.to(device).detach())



def run_cl_epoch(
    device,
    is_master:bool,
    uncond : bool,
    model,
    diffusion,
    options: dict,
    dataloader: th.utils.data.DataLoader,
    optimizer: th.optim.Optimizer,
    prompt,  # prompt for inference, not training
    sample_bs: int,  # batch size for inference
    sample_gs: float = 10.0,  # guidance scale for inference
    sample_respacing: str = '100', # respacing for inference
    outputs_dir: str = "./outputs",
    checkpoints_dir: str = "./checkpoints",
    log_frequency: int = 100,
    wandb_run=None,
    gradient_accumualation_steps=1,
    epoch: int = 0,
    state: int =1,
    energy_mode = False,
    class_id = -1 ,
    clip_grad_norm = False,
    data_name  = "",
):


    train_step = base_train_step

    model.to(device)
    model.train()
    log = {}
    for train_idx, batch in enumerate(dataloader):

        accumulated_loss = train_step(
            model=model,
            diffusion=diffusion,
            batch=batch,
            device=device,
            uncond = uncond,
            learn_sigma=options["learn_sigma"],
        )
        accumulated_loss = accumulated_loss / gradient_accumualation_steps
        accumulated_loss.backward()


        if gradient_accumualation_steps > 1:
            if  (train_idx+1) % gradient_accumualation_steps == 0 or train_idx == len(dataloader) - 1:
                if clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),10)
                optimizer.step()
                model.zero_grad() 
        else:
            if clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(),10)
            optimizer.step()
            model.zero_grad()
  

        if is_master and wandb_run is not None:
            log = {"iter": train_idx, "state": state,"loss": accumulated_loss.item() / gradient_accumualation_steps}


        if is_master and train_idx > 0 and train_idx % log_frequency == 0:
            
       
            print(f"loss: {accumulated_loss.item():.4f}")
            print(f"Sampling from model at iteration {train_idx}")
        
     
            if energy_mode:
                sampler = util.energy_sample
            else:
                sampler = util.sample

            samples =sampler(
                    uncond = uncond,
                    state = state,
                    model=model,
                    options=options,
                    batch_size=sample_bs,
                    guidance_scale=sample_gs, 
                    device=device,
                    prediction_respacing=sample_respacing,
                    class_id = class_id,
                    data_name=data_name
                )
            sample_save_path = os.path.join(outputs_dir, f"{state}_{train_idx}.png")
            train_util.pred_to_pil(samples).save(sample_save_path)

            if wandb_run is not None:
                if uncond:
                        wandb_run.log(
                            {
                                **log,
                                "iter": train_idx,
                                "samples": wandb.Image(sample_save_path, caption="Unconditional"),
                            }
                        )
                else:
                        wandb_run.log(
                                {
                                    **log,
                                    "samples": wandb.Image(sample_save_path,caption=prompt["caption"]),
                                }
                            )
            print(f"Saved sample {sample_save_path}")
        
        if is_master and train_idx % 5000 == 0 and train_idx > 0:
            train_util.save_model(model, checkpoints_dir, train_idx, epoch,state)
            print(
                f"Saved checkpoint {train_idx} to {checkpoints_dir}/model-ft-{train_idx}.pt"
            )
        
        if is_master and wandb_run is not None:
            wandb_run.log(log)
        else:
            print(log)

    if is_master:
     
        print(f"Finished training, saving final checkpoint")
        train_util.save_model(model, checkpoints_dir, train_idx, epoch,state)
    

