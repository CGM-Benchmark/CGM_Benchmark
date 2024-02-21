
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


def get_caption_simple(label):
    shapes_to_idx = {"cube": 0, "sphere": 1,"cylinder":2}

    shapes = list(shapes_to_idx.keys())

    return f'A {shapes[label[0]]}'


parser = argparse.ArgumentParser()


parser.add_argument('--ckpt_path', required=True)
parser.add_argument('--diffusion_steps', type=int, default=100)
parser.add_argument('--noise_schedule', type=str, default='linear',choices=['linear', 'squaredcos_cap_v2'])
parser.add_argument('--sampler', type=str, default="REV",choices=["HMC", "UHMC", "ULA","REV"])
parser.add_argument('--energy_mode', action='store_true')
parser.add_argument('--num_classes',type=str, default='3,')
parser.add_argument('--is_ae', action='store_true')
args = parser.parse_args()



has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')




options = model_and_diffusion_defaults()

#64x64
model_path1= args.ckpt_path # 
options["noise_schedule"]= args.noise_schedule
options["learn_sigma"] = False
options["num_classes"] = args.num_classes  # "3," c-Blender, "6," c-MSCOCO , "10," FOR CIFAR10
options["dataset"] ="cl"
options["image_size"] =   64#  128 , 3 
options["num_channels"] = 128 #192 
options["num_res_blocks"] = 3 #2
options["energy_mode"] = args.energy_mode
options['diffusion_steps'] = args.diffusion_steps
options['is_ae']= args.is_ae

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

# FOR Batch generations of Spheres 
batch_size = 8
labels = th.tensor([ [ [1] ],[ [1] ],[ [1] ],[ [1] ],[ [1] ],[ [1] ],[ [1] ],[ [1] ] ]).long()

# USE BELOW FOR COMPOSITION
# You need to change the batch size to 1 and also needs to change the model_fn to cfg_model_fn in the sampling step below
# batch_size = 1

# labels = th.tensor([[ [0], [2] ]]).long() # Compose Cube And Cylinder Labels
# labels = th.tensor([[ [1], [2] ]]).long() # # Compose Sphere And Cylinder Labels

[print(get_caption_simple(lab.numpy())) for lab in labels[0]]
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


# To generate Batch wise single label conditioned generations this fn can be used
def model_fn(x_t, ts, **kwargs):
    half = x_t[: len(x_t) // 2]
    combined = th.cat([half, half], dim=0)
    if options['energy_mode']:
        eps = model1(combined, ts,eval=True, **kwargs)
    else:
        eps = model1(combined, ts, **kwargs)
    
    masks = kwargs.get('masks')
    cond_eps, uncond_eps = eps[masks], eps[~masks]

    half_eps = uncond_eps +guidance_scale* (cond_eps - uncond_eps) 
    eps = th.cat([half_eps, half_eps], dim=0)
 
    return eps

# For Compositions use this fn
def cfg_model_fn(x_t, ts, **kwargs):
    half = x_t[:1]
    combined = th.cat([half] * kwargs['y'].size(0), dim=0)
    if options['energy_mode']:
        eps = model1(combined, ts,eval=True, **kwargs)
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
# For Custom-Blender : 
num_steps = args.diffusion_steps

la_steps = 20
la_multiplier  =0.00001
la_step_sizes = diffusion.betas*la_multiplier

ha_steps = 10# Hamiltonian steps to run
num_leapfrog_steps = 3 # Steps to run in leapfrog
damping_coeff = 0.9
mass_diag_sqrt = diffusion.betas 

ha_step_sizes = (diffusion.betas) *0.10 

def gradient_cha(x_t, ts, **kwargs):
    half = x_t[:1]
    combined = th.cat([half] * kwargs['y'].size(0), dim=0)
    energy_norm,eps = model1(combined, ts, energy_sampler=True,**kwargs)

    cond_energy,uncond_energy = energy_norm[:-1], energy_norm[-1:]
    total_energy = uncond_energy.sum() + guidance_scale*(cond_energy.sum() - uncond_energy.sum())
    
    
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
                gradient_cha,
                None)

elif args.sampler == 'REV':
    print("Using Reverse Diffusion Sampling only")
    sampler = None
else:
    print("Using Reverse Diffusion Sampling only")
    sampler = None

print("Using Sampler: ",args.sampler)
all_samp = []


for k in range(1):
    if options['energy_mode']:
        samples = diffusion.p_sample_loop(
            sampler,
            model_fn,
            (full_batch_size, 3, 64, 64),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
    else:
        samples = diffusion.p_sample_loop(
            model_fn,
            (full_batch_size, 3, 64, 64),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
            sampler=sampler,
        )[:batch_size]
    sample = samples.contiguous()

    sample = convert_images(sample)

    show_img = sample.cpu().detach().numpy()
    all_samp.append(show_img)

    
fig ,ax = plt.subplots(figsize=(32,32))

arr = np.concatenate(all_samp, axis=0)
show_img = th.tensor(arr)
show_img = show_img.permute(0, 3, 1, 2) # N C H W


w = 4
h = 4
fig = plt.figure(figsize=(32,32))
columns = 4
rows = 2
for i in range(1, columns*rows +1):
    img = show_img[i-1].permute(1,2,0).numpy()
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    plt.axis("off")
    # cap = "A Sphere And A Cylinder"
    cap = "A Sphere"

    plt.title(cap,fontsize=25)


plt.savefig(f"Energy_Object_{cap}_REV_{guidance_scale}.png")
