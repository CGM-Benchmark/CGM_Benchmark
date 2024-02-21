# Description: Sample images from a trained model.

import argparse
import os
from mpi4py import MPI
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from composable_diffusion import dist_util
from PIL import Image
from composable_diffusion.model_creation import Sampler_create_gaussian_diffusion,create_model_and_diffusion,model_and_diffusion_defaults,create_gaussian_diffusion
import argparse 
from torchvision import transforms 
from torch.utils import data
from torch.utils.data import DataLoader



def pil_image_to_norm_tensor(pil_image):
    """
    Convert a PIL image to a PyTorch tensor normalized to [-1, 1] with shape [B, C, H, W].
    """
    return th.from_numpy(np.asarray(pil_image)).float().permute(2, 0, 1) / 127.5 - 1.0


class domain_inc_3states(data.Dataset):
    def __init__(self, 
                shard,
                num_shards,
                state,
                ):
        
        self.labels_1 = [int(state)-1]*500
        self.labels_1 = np.array(self.labels_1)
       


        self.labels_1 = self.labels_1[shard:][::num_shards]

 
        self.size = self.labels_1.shape[0]


        print('label data size', self.labels_1.shape)


    def __len__(self):
        return self.size

    def __getitem__(self, index):
    
        label_1 = self.labels_1[index]
       
        return th.tensor(label_1,dtype=th.long)

def convert_images(batch: th.Tensor):
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).permute(0, 2, 3, 1)

    return scaled


def generate_images(args,model,diffusion,label,guidance_scale=4):
    
    batch_size = label.shape[0]
 
    masks1 = [True] * batch_size + [False] * batch_size

    labels1 = th.cat(([label] + [th.zeros_like(label)]), dim=0)
   
    model_kwargs1 = dict(
    y= th.tensor(labels1,device=dist_util.dev()),
    masks=th.tensor(masks1, dtype=th.bool, device=dist_util.dev())
        )

    full_batch_size = batch_size * 2


    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        eps = model(combined, ts,eval=True, **kwargs)
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        masks = kwargs.get('masks')
        cond_eps, uncond_eps = eps[masks], eps[~masks]
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return eps

    def model_fn_noen(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        eps = model(combined, ts, **kwargs)
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        masks = kwargs.get('masks')
        cond_eps, uncond_eps = eps[masks], eps[~masks]
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return eps


    if args.energy_mode:
        samples = diffusion.p_sample_loop(
                None,
                model_fn,
                (full_batch_size, 3, args.image_size, args.image_size),
                device=dist_util.dev(),
                clip_denoised=True,
                progress=False,
                model_kwargs=model_kwargs1,
                cond_fn=None,
            )[:batch_size]
    else: 
        samples = diffusion.p_sample_loop(
                model_fn_noen,
                (full_batch_size, 3, args.image_size, args.image_size),
                device=dist_util.dev(),
                clip_denoised=True,
                progress=False,
                model_kwargs=model_kwargs1,
                cond_fn=None,
            )[:batch_size]


    return samples


def main(args):

    options = model_and_diffusion_defaults()

    if args.diffusion_steps == 1000:
        base_timestep_respacing = '1000'
    elif args.diffusion_steps == 100:
        base_timestep_respacing = '100'

    options["noise_schedule"]= args.noise_schedule
    options["learn_sigma"] = False
    options["num_classes"] = args.num_classes 
    options["dataset"] = "cl"
    options["image_size"] =  args.image_size  
    options["num_channels"] = 128 #192 
    options["num_res_blocks"] = 3 #2
    options['timestep_respacing'] = base_timestep_respacing # use 100 diffusion steps for fast sampling
    options['energy_mode'] = args.energy_mode
    options['diffusion_steps']= args.diffusion_steps
    options['is_ae'] =args.is_ae


    dist_util.setup_dist()
    

    if options['energy_mode']:
        print("Using energy mode")
        diffusion = Sampler_create_gaussian_diffusion(
        steps=args.diffusion_steps,
        learn_sigma=options['learn_sigma'],
        noise_schedule=options['noise_schedule'],
        timestep_respacing=base_timestep_respacing,
        )
    else:
        diffusion = create_gaussian_diffusion(
        steps=args.diffusion_steps,
        learn_sigma=options['learn_sigma'],
        noise_schedule=options['noise_schedule'],
        timestep_respacing=base_timestep_respacing,
        )
    if len(args.model_path) > 0:
        assert os.path.exists(
            args.model_path
        ), f"Failed to resume from {args.model_path}, file does not exist."
        weights = th.load(args.model_path, map_location="cpu")
        model, _ = create_model_and_diffusion(**options)
        model.load_state_dict(weights)
        print(f"Resumed from {args.model_path} successfully.")

    model.to(dist_util.dev())
    model.eval()



    print( 'Rank',MPI.COMM_WORLD.Get_rank())
    print("World size:", MPI.COMM_WORLD.Get_size())


    dataset = domain_inc_3states(
                shard = MPI.COMM_WORLD.Get_rank(),
                num_shards = MPI.COMM_WORLD.Get_size(),
                state = args.state,
        )

    dataloader = DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers = args.num_workers)
    

    all_images = []
    all_labels = []
    all_ori_imgs = []

    for i,label in enumerate(dataloader):
        label = label.to(dist_util.dev())
        
        sample = generate_images(args,model,diffusion,label,guidance_scale=args.guidance_scale)
        
        sample = sample.contiguous()
        sample = convert_images(sample).contiguous()
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        label = label.contiguous()
        gathered_labels = [th.zeros_like(label) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, label)

        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

        

   
    arr = np.concatenate(all_images, axis=0)
    label_arr = np.concatenate(all_labels, axis=0)
 
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        if args.energy_mode:
            out_path = os.path.join(f"./{args.data_name}_{args.guidance_scale}_{args.state}_{shape_str}_energy.npz")
        else:
            out_path = os.path.join(f"./{args.data_name}_{args.guidance_scale}_{args.state}_{shape_str}.npz")
        print(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    dist.barrier()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--state', required=True,type = int)
    parser.add_argument('--num_workers',default=2,type = int)
    parser.add_argument('--data_path',default="./",type = str)
    parser.add_argument('--eval_idx_path',default="./",type = str)
    parser.add_argument('--image_size',default=64,type = int)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--guidance_scale", type=float, default=12.0)
    parser.add_argument('--batch_size',type=int,default=256)
    parser.add_argument('--data_name',type=str,default='c-domain',choices=['c-domain'])
    parser.add_argument('--noise_schedule',type=str,default='linear',choices=['linear','squaredcos_cap_v2'])
    parser.add_argument('--diffusion_steps', type=int, default=100)
    parser.add_argument('--energy_mode', action='store_true')
    parser.add_argument( "--num_classes",  type=str, default="" )
    parser.add_argument('--is_ae', action='store_true')
    args = parser.parse_args()


    
    main(args)