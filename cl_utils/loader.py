import time
from pathlib import Path

import random
import PIL
from PIL import Image

import torch as th
from torch.utils.data import Dataset
from torchvision import transforms as T
from cl_utils.train_util import pil_image_to_norm_tensor
import numpy as np


from torch.utils import data
import os 

def random_resized_crop(image, shape, resize_ratio=1.0):
    """
    Randomly resize and crop an image to a given size.

    Args:
        image (PIL.Image): The image to be resized and cropped.
        shape (tuple): The desired output shape.
        resize_ratio (float): The ratio to resize the image.
    """
    image_transform = T.RandomResizedCrop(shape, scale=(resize_ratio, 1.0), ratio=(1.0, 1.0))
    return image_transform(image)

def get_prompt_name(id):
    "Get Dictionary of mapping Custom MSCOCO labels to class names"
    label_to_class = {
        0: "cat",
        1: "bed",
        2: "couch",
        3: "tv",
        4: "pizza",
        5: "dining table",
    }

    return label_to_class[id]

def get_prompt_name_domain(id):
    "Get Dictionary of mapping Custom Domain labels to class names"
    label_to_class = {
        0: "bedroom",
        1: "classroom",
        2: "conference_room",
        3: "kitchen",
        4: "tower",
        5: "black widow",
        6: "ruffed grouse",
        7: "toucan",
        8: "tiger",
        9: "zucchini",
    }

    return label_to_class[id]


class COCO_64_cl(Dataset):
    def __init__(self, state,
    data_path ="./",
    buffer = True , 
    uncond_p = 0.05,
           ):

        print("Loading data from ", data_path)
        data = np.load(data_path)

        if state == 1:
            ims = data['state1_img']
            labels = data['state1_lab']

        elif state==2:
        
            ims_2 = list(data['state2_img'])
            labels_2 = list(data['state2_lab'])
            
            buffer_state2_img = list(buffer['state2_buff_img'])
            buffer_state2_lab = list(buffer['state2_buff_lab'])

            ims_2.extend(buffer_state2_img)
            labels_2.extend(buffer_state2_lab)

            ims = ims_2
            labels = labels_2

        elif state==3:

            ims_3 = list(data['state3_img'])
            labels_3 = list(data['state3_lab'])

            buffer_state3_img = list(buffer['state3_buff_img'])
            buffer_state3_lab = list(buffer['state3_buff_lab'])

            ims_3.extend(buffer_state3_img)
            labels_3.extend(buffer_state3_lab)

            ims = ims_3
            labels = labels_3



        self.ims = np.array(ims)
        self.labels = np.array(labels)
        self.labels=self.labels.reshape(self.labels.shape[0],)

        self.uncond_p =  uncond_p
        self.size = self.labels.shape[0]

    
        print(f"size of the data : {self.ims.shape}, {self.labels.shape}")



    def __len__(self):
        return self.size

    def __getitem__(self, index):
    
        im = Image.fromarray(self.ims[index])
        label = self.labels[index]
       
        mask = random.random() > self.uncond_p

        base_tensor =pil_image_to_norm_tensor(im)

        return  th.tensor(label,dtype=th.long),th.tensor(mask, dtype=th.bool), base_tensor
    
    def get_test_sample(self, state):
        if state==1:
            label_prompt = 1 
        elif state==2:
            label_prompt = 2
        elif state==3:
            label_prompt = 5

        test_prompt= get_prompt_name(label_prompt)
        test_prompt_lab= {"caption":test_prompt,"label":th.tensor(label_prompt,dtype=th.long)}

        return (test_prompt_lab,label_prompt)



class blender_cl_64(data.Dataset):
    def __init__(self,
                data_path, 
                state,
                buffer =True,
                uncond_p = 0.05,
                ):

        data_blender= np.load(data_path)

        if state ==1:
            self.ims_1 = data_blender["task1_img"]
            self.labels_1 = np.array(data_blender["task1_lab"])
            
        if state ==2:
       
            self.ims_1 = data_blender["task2_img"]
            self.labels_1 = np.array(data_blender["task2_lab"])

            self.ims_1 = list(self.ims_1)
            self.labels_1 = list(self.labels_1)
                
            # Buffer replay from state 1 
            if buffer:
                indexes = data_blender['buffer_idx_task_2']
                
                self.buffer_img_1 = list(data_blender['task1_data'][indexes])
                self.buffer_labels_1 = list(data_blender['task1_lab'][indexes])
            
                self.ims_1.extend(self.buffer_img_1)
                self.labels_1.extend(self.buffer_labels_1)


        if state ==3:
            self.ims_1 =data_blender["task3_img"]
            self.labels_1 = np.array(data_blender["task3_lab"])
            
            
            self.ims_1 = list(self.ims_1)
            self.labels_1 = list(self.labels_1)


            if buffer:
                indexes = data_blender['buffer_idx_task_3']
             
                
                
                self.buffer_img_1 = list(data_blender['task1_data'][indexes])
                self.buffer_labels_1 = list(data_blender['task1_lab'][indexes])

                self.ims_1.extend(self.buffer_img_1)
                self.labels_1.extend(self.buffer_labels_1)

                self.buffer_img_2 = list(data_blender['task2_data'][indexes])
                self.buffer_labels_2 =list(data_blender['task2_lab'][indexes])
               
                self.ims_1.extend(self.buffer_img_2)
                self.labels_1.extend(self.buffer_labels_2)

   
        self.ims_1 = np.array(self.ims_1)
        self.labels_1 = np.array(self.labels_1)

        self.shapes_to_idx = {"cube": 0, "sphere": 1,"cylinder":2}
      

        self.shapes = list(self.shapes_to_idx.keys())

        # Randomly mask % of labels during training
        self.uncond_p = uncond_p
        self.size = self.labels_1.shape[0]


        print('image data size', self.ims_1.shape)
        print('label data size', self.labels_1.shape)


    def __len__(self):
        return self.size

    def __getitem__(self, index):
    
        im_1 = Image.fromarray(self.ims_1[index])
        label_1 = self.labels_1[index]

        mask = random.random() > self.uncond_p
        base_tensor = pil_image_to_norm_tensor(im_1)
   
        return  th.tensor(label_1,dtype=th.long),th.tensor(mask, dtype=th.bool), base_tensor

    def get_test_sample(self,state):
        
        if state==1:
            label = [0]
        elif state==2:
            label = [1]
        elif state==3:
            label = [2]

        description = self._convert_caption(label).strip()
        print(f"The label:{label} corresponding to the caption: {description}")

        return {"caption":description,"label":th.tensor(label,dtype=th.long)}

  
    def _convert_caption(self, label):


        return f'A {self.shapes[label[0]]}'
      
class domain_cl(Dataset):
    def __init__(self,
                state,
                buffer=True,
                data_path ="/domain_cl_train_data.npz",
                uncond_p = 0.05,
                ):

        print("Loading data from ", data_path)
        data = np.load(data_path)

        
        if state == 1:
            ims = data['task1_img']
            labels = data['task1_lab']

        elif state==2:
        
            ims_2 = list(data['task2_img'])
            labels_2 = list(data['task_2_lab'])
            
            if buffer:
                buffer_state2_img = list(data['buff_images'])
                buffer_state2_lab = list(data['buff_labels'])

                ims_2.extend(buffer_state2_img)
                labels_2.extend(buffer_state2_lab)

            ims = ims_2
            labels = labels_2



        self.ims = np.array(ims)
        self.labels = np.array(labels)
        self.labels=self.labels.reshape(self.labels.shape[0],)
        print(f"size of the data : {self.ims.shape}, {self.labels.shape}")

        print(np.unique(self.labels, return_counts=True))
        self.uncond_p = uncond_p
        self.size = self.labels.shape[0]


        print('image data size', self.ims.shape)
        print('label data size', self.labels.shape)


    def __len__(self):
        return self.size

    def __getitem__(self, index):
    
        im = Image.fromarray(self.ims[index])
        label = self.labels[index]
       
        mask = random.random() > self.uncond_p
        base_tensor =pil_image_to_norm_tensor(im)

        return  th.tensor(label,dtype=th.long),th.tensor(mask, dtype=th.bool), base_tensor

    def get_test_sample(self, state):
        if state==1:
            label_prompt = 0 # Task 1 is LSUN - 5classes
        elif state==2:
            label_prompt = 5 # Task 2 is Imagenet - 5classes 


        test_prompt= get_prompt_name_domain(label_prompt)
        test_prompt_lab= {"caption":test_prompt,"label":th.tensor(label_prompt,dtype=th.long)}

        return (test_prompt_lab,label_prompt)