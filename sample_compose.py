
import torch as th
import matplotlib.pyplot as plt

import numpy as np 
import os 
from composable_diffusion.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    create_gaussian_diffusion,
    Sampler_create_gaussian_diffusion,
)

from composable_diffusion.model_creation import create_model_and_diffusion,model_and_diffusion_defaults,create_gaussian_diffusion

from anneal_samplers import AnnealedCHASampler, AnnealedUHASampler,AnnealedULASampler
import argparse


def pil_image_to_norm_tensor(pil_image):
    """
    Convert a PIL image to a PyTorch tensor normalized to [-1, 1] with shape [B, C, H, W].
    """
    return th.from_numpy(np.asarray(pil_image)).float().permute(2, 0, 1) / 127.5 - 1.0


def convert_images(batch: th.Tensor):
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).permute(0, 2, 3, 1)

    return scaled


parser = argparse.ArgumentParser()


parser.add_argument('--ckpt_path', required=True)
parser.add_argument('--diffusion_steps', type=int, default=100)
parser.add_argument('--noise_schedule', type=str, default='linear',choices=['linear', 'squaredcos_cap_v2'])
parser.add_argument('--sampler', type=str, default="UHMC",choices=["HMC", "UHMC", "ULA","Rev_Diff"])
parser.add_argument('--data_name',default='c-blender',choices=['c-blender','c-mscoco'])
parser.add_argument('--energy_mode', action='store_true')
parser.add_argument('--state', type=int, default=0)
parser.add_argument('--image_size',type=int, default=64)
parser.add_argument('--num_samples',type=int, default=100)
parser.add_argument('--num_classes',type=str, default='3,')
args = parser.parse_args()



has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')




options = model_and_diffusion_defaults()

#64x64
model_path1= args.ckpt_path # 
options["noise_schedule"]= args.noise_schedule
options["learn_sigma"] = False
options["num_classes"] = args.num_classes  # "4,"
options["dataset"] ="cl"
options["image_size"] =   args.image_size#  128 , 3 
options["num_channels"] = 128 #192 
options["num_res_blocks"] = 3 #2
options["energy_mode"] = args.energy_mode
options['diffusion_steps'] = args.diffusion_steps


if args.diffusion_steps == 100:
    base_timestep_respacing = '100' 
elif args.diffusion_steps == 1000:
    base_timestep_respacing = '1000'



if options['energy_mode']:
    print("Using energy mode")
    diffusion = Sampler_create_gaussian_diffusion(
    steps=options['diffusion_steps'],
    learn_sigma=options['learn_sigma'],
    noise_schedule=options['noise_schedule'],
    timestep_respacing=base_timestep_respacing,
    )
else:
    diffusion = create_gaussian_diffusion(
    steps=options['diffusion_steps'],
    learn_sigma=options['learn_sigma'],
    noise_schedule=options['noise_schedule'],
    timestep_respacing=base_timestep_respacing,
    )

if len(model_path1) > 0:
    assert os.path.exists(
        model_path1
    ), f"Failed to resume from {model_path1}, file does not exist."
    weights = th.load(model_path1, map_location="cpu")
    model1,_ = create_model_and_diffusion(**options)
    model1.load_state_dict(weights)


model1 = model1.to("cuda")
model1.eval()



guidance_scale = 4
batch_size = 1


curr_state = args.state

if args.data_name == 'c-blender':
    if curr_state==1:
        labels = th.tensor([[ [0], [1] ]]).long()
    elif curr_state==2:
        labels = th.tensor([[ [1], [2] ]]).long()
    elif curr_state==3:
        labels = th.tensor([[ [0], [2] ]]).long()
    elif curr_state==4:
        labels = th.tensor([[ [0],[1], [2] ]]).long()
elif args.data_name == 'c-mscoco':
    if curr_state==1:
        labels = th.tensor([[ [0], [1] ]]).long()
    elif curr_state==2:
        labels = th.tensor([[ [2], [3] ]]).long()
    elif curr_state==3:
        labels = th.tensor([[ [4], [5] ]]).long()



print(labels)
print(labels.shape)



labels = [x.squeeze(dim=1) for x in th.chunk(labels, labels.shape[1], dim=1)]
full_batch_size = batch_size * (len(labels) + 1)

masks = [True] * len(labels) * batch_size + [False] * batch_size

labels = th.cat((labels + [th.zeros_like(labels[0])]), dim=0)

model_kwargs = dict(
    y=labels.clone().detach().to(device),
    masks=th.tensor(masks, dtype=th.bool, device=device)
)
print(labels)
print(masks)
print(labels.shape)



def model_fn_t(x_t, ts, **kwargs):
    cond_eps = model1(x_t, ts,eval=True, **kwargs)
    kwargs['y'] = th.zeros(kwargs['y'].shape, dtype=th.long,device = "cuda")
    kwargs['masks'] = th.tensor([False] * batch_size, dtype=th.bool, device=device)
    uncond_eps = model1(x_t, ts,eval=True, **kwargs)

    eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)

    return eps


def cfg_model_fn(x_t, ts, **kwargs):
    half = x_t[:1]
    combined = th.cat([half] * kwargs['y'].size(0), dim=0)
    if options['energy_mode']:
        eps = model1(combined, ts, eval=True,**kwargs)
    else:
        eps = model1(combined, ts,**kwargs)
    cond_eps, uncond_eps = eps[:-1], eps[-1:]
    # assume weights are equal to guidance scale

    half_eps = uncond_eps + guidance_scale*(cond_eps - uncond_eps).sum(dim=0, keepdim=True)
    eps = th.cat([half_eps] * x_t.size(0), dim=0)
    return eps


alphas = 1 - diffusion.betas
alphas_cumprod = np.cumprod(alphas)
scalar = np.sqrt(1 / (1 - alphas_cumprod))

def gradient(x_t, ts, **kwargs):
    half = x_t[:1]
    combined = th.cat([half] * kwargs['y'].size(0), dim=0)
    if options['energy_mode']:
        eps = model1(combined, ts, eval=True,**kwargs)
    else:
        eps = model1(combined, ts,**kwargs)
    cond_eps, uncond_eps = eps[:-1], eps[-1:]
    # assume weights are equal to guidance scale
    half_eps = uncond_eps + guidance_scale*(cond_eps - uncond_eps).sum(dim=0, keepdim=True)
    eps = th.cat([half_eps] * x_t.size(0), dim=0)  
    # Need to scale the gradients by coefficient to properly account for normalization in DSM loss + data contraction
    scale = scalar[ts[0]]
    return -1*scale*eps

# Hypeprparameters For Samplers : Need to be tuned carefully for generating good proposals

num_steps = args.diffusion_steps

#ULA 
# increase the number of Langevin MCMC steps run to sample between intermediate distributions
# more steps improves sampling quality
la_steps = 20
la_step_sizes = diffusion.betas * 2


#HMC / UHMC SAMPLER
ha_steps = 10#2 # Hamiltonian steps to run
num_leapfrog_steps = 3 # Steps to run in leapfrog
damping_coeff = 0.7#0.9
mass_diag_sqrt = diffusion.betas
ha_step_sizes = (diffusion.betas) * 0.1 #0.1



def gradient_cha(x_t, ts, **kwargs):
    half = x_t[:1]
    combined = th.cat([half] * kwargs['y'].size(0), dim=0)
    energy_norm,eps = model1(combined, ts, energy_sampler=True,**kwargs)
    
    cond_energy,uncond_energy = energy_norm[:-1], energy_norm[-1:]
    total_cond = cond_energy.sum(axis=0).unsqueeze(0)

    total_energy = uncond_energy.sum() + guidance_scale*(total_cond.sum() - uncond_energy.sum())
    
    cond_eps, uncond_eps = eps[:-1], eps[-1:]
    # assume weights are equal to guidance scale
    half_eps = uncond_eps +guidance_scale* (cond_eps - uncond_eps).sum(dim=0, keepdim=True)
    eps = th.cat([half_eps] * x_t.size(0), dim=0)  

    # Need to scale the gradients by coefficient to properly account for normalization in DSM loss + data contraction
    # print(ts)
    scale = scalar[ts[0]]
    return -scale*total_energy,-1*scale*eps



if args.sampler == 'ULA':
    sampler = AnnealedULASampler(num_steps, la_steps, la_step_sizes, gradient)
elif args.sampler == 'UHMC':
    sampler = AnnealedUHASampler(num_steps,
                ha_steps,
                ha_step_sizes,
                damping_coeff,
                mass_diag_sqrt,
                num_leapfrog_steps,
                gradient,
                )
elif args.sampler == 'HMC':
    sampler = AnnealedCHASampler(num_steps,
                ha_steps,
                ha_step_sizes,
                damping_coeff,
                mass_diag_sqrt,
                num_leapfrog_steps,
                gradient_cha)

elif args.sampler == 'Rev_Diff':
    print("Using Reverse Diffusion Sampling only")
    sampler = None

print("Using Sampler: ",args.sampler)
all_samp = []


for k in range(args.num_samples):
    if options['energy_mode']:
        samples = diffusion.p_sample_loop(
            sampler,
            cfg_model_fn,
            (full_batch_size, 3, args.image_size, args.image_size),
            device=device,
            clip_denoised=True,
            progress=False,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
    else:
        samples = diffusion.p_sample_loop(
            cfg_model_fn,
            (full_batch_size, 3, args.image_size, args.image_size),
            device=device,
            clip_denoised=True,
            progress=False,
            model_kwargs=model_kwargs,
            cond_fn=None,
            sampler=sampler,
        )[:batch_size]
    sample = samples.contiguous()

    sample = convert_images(sample)

    show_img = sample.cpu().detach().numpy()
    all_samp.append(show_img)

arr = np.concatenate(all_samp, axis=0)

np.savez(f'{args.data_name}_compose_{args.state}_{args.num_samples}_samples.npz', arr)
print(" Done Generations !! ")
