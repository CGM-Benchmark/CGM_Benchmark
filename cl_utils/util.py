## util.py

from typing import Tuple

import PIL
import numpy as np
import torch as th
from composable_diffusion.model_creation import (
    create_gaussian_diffusion,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
    Sampler_create_gaussian_diffusion
)

MODEL_TYPES = ["base", "upsample", "base-inpaint", "upsample-inpaint"]



def load_model(
    diffusion_steps: int,
    energy_mode: bool,
    learn_sigma: bool,
    num_classes: str = "",
    model_type: str = "base",
    noise_schedule: str = "squaredcos_cap_v2",
    dropout: float = 0.0,
    data_name = "",
):
    assert model_type in MODEL_TYPES, f"Model must be one of {MODEL_TYPES}. Exiting."
    if model_type in ["base", "base-inpaint"]:
        options = model_and_diffusion_defaults()
    elif model_type in ["upsample", "upsample-inpaint"]:
        options = model_and_diffusion_defaults_upsampler()
    if "inpaint" in model_type:
        options["inpaint"] = True

    if data_name=="cifar10":
        options["noise_schedule"]= noise_schedule
        options["learn_sigma"] = learn_sigma
        options["num_classes"] = None if num_classes == "" else num_classes
        options["dataset"] = "cl"
        options["image_size"] = 32 
        options["num_channels"] = 128 
        options["num_res_blocks"] = 3
        options["dropout"] = dropout
        options["energy_mode"] = energy_mode
        options["diffusion_steps"] = diffusion_steps
    elif data_name=="c-mscoco":
        options["noise_schedule"]= noise_schedule
        options["learn_sigma"] = learn_sigma
        options["num_classes"] = None if num_classes == "" else num_classes
        options["dataset"] = "cl"
        options["image_size"] = 64
        options["num_channels"] = 128 # 192
        options["num_res_blocks"] =3
        options["dropout"] = dropout      
        options["energy_mode"] = energy_mode
        options["diffusion_steps"] = diffusion_steps  
    elif data_name=="c-blender":
        options["noise_schedule"]= noise_schedule
        options["learn_sigma"] = learn_sigma
        options["num_classes"] = None if num_classes == "" else num_classes
        options["dataset"] = "cl"
        options["image_size"] =  64  # 64 , 128 , 3 
        options["num_channels"] = 128#128 #192 
        options["num_res_blocks"] = 3#3 #2
        options["energy_mode"] = energy_mode
        options["diffusion_steps"] = diffusion_steps

    print("Using Energy Based Model?: " , energy_mode)
    print(options)


    model, diffusion = create_model_and_diffusion(**options)

    return model, diffusion, options

def read_image(path: str, shape: Tuple[int, int]):
    pil_img = PIL.Image.open(path).convert('RGB')
    pil_img = pil_img.resize(shape, resample=PIL.Image.BICUBIC)
    img = np.array(pil_img)
    return th.from_numpy(img)[None].permute(0, 3, 1, 2).float() / 127.5 - 1



# Sample from the model
@th.inference_mode()
def sample(
    state,
    model,
    options,
    uncond=False,
    batch_size=1,
    guidance_scale=4,
    device="cpu",
    prediction_respacing="100",
    class_id = -1 ,  # -1 means no task id
    data_name = "",
):
    eval_diffusion = create_gaussian_diffusion(
        steps=options["diffusion_steps"],
        noise_schedule=options["noise_schedule"],
        timestep_respacing=prediction_respacing,
        learn_sigma=options["learn_sigma"],
    )

    if data_name=="cifar10":
        labels = th.tensor([[int(class_id)]]).long()
        size = 32
    elif data_name=="c-MSCOCO":
        labels = th.tensor([[int(class_id)]]).long()
        size = 64 
    elif data_name=="c-blender":
        size = 64
        if state==1:
            labels = th.tensor([[0]]).long()
        elif state==2:
            labels = th.tensor([[1]]).long()
        elif state==3:
            labels = th.tensor([[2]]).long()

    full_batch_size = batch_size * (len(labels) + 1)
    masks = [True] * len(labels) + [False]
    labels = th.cat(([labels] + [th.zeros_like(labels)]), dim=0)


    if uncond:
        model_kwargs = dict(
        y=None,
        masks=None
            )
        full_batch_size = batch_size
        def model_fn(x, t, **kwargs):
            return model(x, t, y=None,masks=None)
    else:
        model_kwargs = dict(
        y=labels.clone().detach().to(device),
        masks=th.tensor(masks, dtype=th.bool, device=device)
            )

        def model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = th.cat([half, half], dim=0)
            eps = model(combined, ts, **kwargs)
            masks = kwargs.get('masks')
            cond_eps, uncond_eps = eps[masks], eps[~masks]
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = th.cat([half_eps, half_eps], dim=0)
            return eps


    samples = eval_diffusion.p_sample_loop(
        model_fn,
        (full_batch_size, 3, size, size),  
        device=device,
        clip_denoised=True,
        progress=False,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:batch_size]
    model.train()
    return samples



# @th.inference_mode()
def energy_sample(
    state,
    model,
    options,
    uncond=False,
    batch_size=1,
    guidance_scale=4,
    device="cpu",
    prediction_respacing="100",
    class_id = -1 ,  # -1 means no task id
    data_name = ""
):
    eval_diffusion = Sampler_create_gaussian_diffusion(
        steps=options["diffusion_steps"],
        noise_schedule=options["noise_schedule"],
        timestep_respacing=prediction_respacing,
        learn_sigma=options["learn_sigma"],
    )

    model.eval()


    if data_name == "cifar10":
        # CIFAR10
        labels = th.tensor([[int(class_id)]]).long()
        size = 32 
    elif data_name =="c-MSCOCO":
        labels = th.tensor([[int(class_id)]]).long()
        size = 64 
    elif data_name=="c-blender":
        size = 64
        if state==1:
            labels = th.tensor([[0]]).long()
        elif state==2:
            labels = th.tensor([[1]]).long()
        elif state==3:
            labels = th.tensor([[2]]).long()


    full_batch_size = batch_size * (len(labels) + 1)
    masks = [True] * len(labels) + [False]
    labels = th.cat(([labels] + [th.zeros_like(labels)]), dim=0)


    if uncond:
        model_kwargs = dict(
        y=None,
        masks=None
            )
        full_batch_size = batch_size
        def model_fn(x, t, **kwargs):
            return model(x, t, eval=True,y=None,masks=None)
    else:
        model_kwargs = dict(
        y=labels.clone().detach().to(device),
        masks=th.tensor(masks, dtype=th.bool, device=device)
            )

        def model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = th.cat([half, half], dim=0)
            eps = model(combined, ts,eval=True, **kwargs)
            masks = kwargs.get('masks')
            cond_eps, uncond_eps = eps[masks], eps[~masks]
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = th.cat([half_eps, half_eps], dim=0)
            return eps



    samples = eval_diffusion.p_sample_loop(
        None, # No MCMC sampler used ; Default : Ancestral Sampler
        model_fn,
        (full_batch_size, 3, size, size),  
        device=device,
        clip_denoised=True,
        progress=False,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:batch_size]
    model.train()
    return samples
