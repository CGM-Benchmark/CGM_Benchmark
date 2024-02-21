import os
import random
import argparse

import torch
import numpy as np

from PIL import Image
from tqdm import tqdm


from torch.utils.data import Dataset, DataLoader
from model import ResNetModel

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]




class MSCOCO_64(Dataset):
    def __init__(
        self,
        resolution,
        npy_path,
        state,
    ):
        self.resolution = resolution
        self.npy_path = npy_path
        self.state = state

        data = np.load(npy_path)

        if npy_path.split(".")[-1]=="npy":
            self.ims = data
        else:
            self.ims = data['arr_0']#np.array(self.ims_1)
            self.labels =data['arr_1']    #np.array(self.labels_1)

            self.labels=self.labels.reshape(self.labels.shape[0],)
        
            print(f"size of the data : {self.ims.shape}, {self.labels.shape}")



    def __len__(self):
        return len(self.ims)
    
    def __getitem__(self, index):
        im = self.ims[index]
        img = Image.fromarray(im).convert('RGB')


        if self.npy_path.split(".")[-1]=="npy":
            if self.state ==1:
                gt_label = np.array([[0,1]])
                
            elif self.state ==2:
                gt_label = np.array([[2,3]])
                
            elif self.state ==3:
                gt_label = np.array([[4,5]])
                
        else:
            gt_label = self.labels[index]


        arr = center_crop_arr(img, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]
        # range 0 to 1
        arr = arr.astype(np.float32) / 255.

        return np.transpose(arr, [2, 0, 1]), gt_label       
     
class c_blender_dataset(Dataset):
    def __init__(
        self,
        resolution,
        npy_path,
        state,
    ):
        self.resolution = resolution
        self.npy_path = npy_path
        self.state = state

        data = np.load(npy_path)

        if npy_path.split(".")[-1]=="npy":
            self.ims = data

        else:
            self.ims = data['arr_0']#np.array(self.ims_1)
            self.labels =data['arr_1']    #np.array(self.labels_1)

            self.labels=self.labels.reshape(self.labels.shape[0],)
        
            print(f"size of the data : {self.ims.shape}, {self.labels.shape}")



    def __len__(self):
        return len(self.ims)
    
    def __getitem__(self, index):
        im = self.ims[index]
        img = Image.fromarray(im).convert('RGB')
        if index==0:
            print(img)
            img.save('testing.png')
            
        if self.npy_path.split(".")[-1]=="npy":
            if self.state ==1:
                gt_label = np.array([[0,1]])
                
            elif self.state ==2:
                gt_label = np.array([[1,2]])
                
            elif self.state ==3:
                gt_label = np.array([[0,2]])
                
            elif self.state ==4:
                gt_label = np.array([[0,1,2]])

        else:
            gt_label = self.labels[index]


        arr = center_crop_arr(img, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]
        # range 0 to 1
        arr = arr.astype(np.float32) / 255.
        # return arr, gt_label
        return np.transpose(arr, [2, 0, 1]), gt_label


def load_classifier(checkpoint_path, dataset, im_size, filter_dim, attr=None):

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f'loading from the latest checkpoint: {checkpoint_path}')

    kwargs = dict(
        spec_norm=True,
        norm=True,
        dataset=dataset,
        lr=1e-5,
        filter_dim=filter_dim,
        im_size=im_size
    )

    model = ResNetModel(**kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model.eval()




def compute_mscoco_classification_score(classifier, npy_path,  image_size, mode,state):
    dataset = MSCOCO_64(npy_path=npy_path, resolution=image_size, state=state)
    dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=32, drop_last=False, num_workers=0, pin_memory=True)

    total_corrects, total_ims = 0, 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for i, (gen_ims, gt_labels) in enumerate(tqdm(dataloader)):
        gen_ims = gen_ims.float().to(device)
        gt_labels = gt_labels.to(device)

        if len(gt_labels.shape) == 3:
            gt_labels = gt_labels.to(device).permute(0,2,1)
            labels = torch.chunk(gt_labels, chunks=gt_labels.shape[1], dim=1)
            labels = [chunk.squeeze(dim=1) for chunk in labels]
        else:
            labels = [gt_labels]

        result = torch.zeros((gen_ims.shape[0]), dtype=torch.long, device=device)
        for label in labels:
            with torch.no_grad():
                outputs = classifier(gen_ims, label)

                # pdb.set_trace()
                result += (outputs[:,0] < outputs[:,1]).long()

        corrects = torch.sum(result == len(labels))

        total_corrects += corrects.item()
        total_ims += gen_ims.shape[0]

    print(f'classification scores: ', (total_corrects / total_ims)*100)

def compute_c_blender_classification_score(classifier, npy_path,  image_size, mode,state):
    dataset = c_blender_dataset(npy_path=npy_path, resolution=image_size,state=state)
    dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=32, drop_last=False, num_workers=0, pin_memory=True)

    total_corrects, total_ims = 0, 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for i, (gen_ims, gt_labels) in enumerate(tqdm(dataloader)):
        gen_ims = gen_ims.float().to(device)
        gt_labels = gt_labels.to(device)
       
        if len(gt_labels.shape) == 3:
            gt_labels = gt_labels.to(device).permute(0,2,1)
            labels = torch.chunk(gt_labels, chunks=gt_labels.shape[1], dim=1)
            labels = [chunk.squeeze(dim=1) for chunk in labels]

        else:
            labels = [gt_labels]

        result = torch.zeros((gen_ims.shape[0]), dtype=torch.long, device=device)
        for label in labels:
            with torch.no_grad():
                outputs = classifier(gen_ims, label)
                result += (outputs[:,0] < outputs[:,1]).long()


        corrects = torch.sum(result == len(labels))

        total_corrects += corrects.item()
        total_ims += gen_ims.shape[0]

    print(f'classification scores: ', (total_corrects / total_ims)*100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # classifier flag
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset", choices=['c-blender','c-mscoco'])
    parser.add_argument("--checkpoint_path", type=str)

    # input images
    parser.add_argument("--im_size", type=int, default=128)
    parser.add_argument("--npy_path", type=str)
    parser.add_argument("--mode", choices=['generation'])
    parser.add_argument("--state", type=int,default=-1)
    # model
    parser.add_argument("--filter_dim", type=int, default=64)

    args = parser.parse_args()
    print(args.npy_path)


    if args.dataset == 'c-blender':
        classifier = load_classifier(checkpoint_path=args.checkpoint_path, dataset=args.dataset,
                                     im_size=args.im_size, filter_dim=args.filter_dim)
        compute_c_blender_classification_score(
            classifier=classifier, npy_path=args.npy_path,
            image_size=args.im_size, mode=args.mode,state=args.state
        )
    elif args.dataset == 'c-mscoco':
        classifier = load_classifier(checkpoint_path=args.checkpoint_path, dataset=args.dataset,
                                     im_size=args.im_size, filter_dim=args.filter_dim)
        compute_mscoco_classification_score(
            classifier=classifier, npy_path=args.npy_path,
            image_size=args.im_size, mode=args.mode,state=args.state
        )
    else:
        raise NotImplementedError


