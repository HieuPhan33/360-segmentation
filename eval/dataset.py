# Code with dataset loader
# April 2019
# Kailun Yang
#######################

import numpy as np
import os
import pandas as pd
import torch

from PIL import Image

from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.png', '.JPG']

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)
    #return filename.endswith("color.png")


def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

def colormap_mapillary(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)
    cmap[0,:] = np.array([153,153,153])
    cmap[1,:] = np.array([210,170,100])
    cmap[2,:] = np.array([220,220,220])
    cmap[3,:] = np.array([250,170, 30])
    cmap[4,:] = np.array([  0,  0,142])
    cmap[5,:] = np.array([  0,  0, 70])

    cmap[6,:] = np.array([119, 11, 32])
    cmap[7,:] = np.array([  0,  0,230])
    cmap[8,:] = np.array([  0, 60,100])
    cmap[9,:] = np.array([220,220,  0])
    cmap[10,:]= np.array([192,192,192])

    cmap[11,:]= np.array([128, 64,128])
    cmap[12,:]= np.array([244, 35,232])
    cmap[13,:]= np.array([170,170,170])
    cmap[14,:]= np.array([140,140,200])
    cmap[15,:]= np.array([128, 64,255])

    cmap[16,:]= np.array([196,196,196])
    cmap[17,:]= np.array([190,153,153])
    cmap[18,:]= np.array([102,102,156])
    cmap[19,:]= np.array([ 70, 70, 70])

    cmap[20,:]= np.array([220, 20, 60])
    cmap[21,:]= np.array([255,  0,  0])
    cmap[22,:]= np.array([ 70,130,180])
    cmap[23,:]= np.array([107,142, 35])
 
    cmap[24,:]= np.array([152,251,152])
    cmap[25,:]= np.array([255,255,255])
    cmap[26,:]= np.array([200,128,128])
    cmap[27,:]= np.array([  0,  0,  0])
    
    return cmap

class cityscapes(Dataset):

    def __init__(self, root, input_transform=None, subset='val'):
        self.images_root = os.path.join(root, 'leftImg8bit/' + subset)
        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.color_map = colormap_mapillary()
       

    def __getitem__(self, index):
        filename = self.filenames[index]
       
        with open(filename, 'rb') as f:
            image = Image.open(f).convert('RGB')
        if self.input_transform is not None:
            image = self.input_transform(image)
        filename = os.path.split(filename)[-1]
        filename = os.path.splitext(filename)[0]
        return image, filename

    def __len__(self):
        return len(self.filenames)

class footpathLabel(Dataset):

    def __init__(self, root, input_transform=None, split=2, subset='val', ignore_index=-1):

        self.images_root = os.path.join(root, f'Split{split}')
        with open(f'{root}/{subset}{split}.txt') as f:
            self.filenames = [line.rstrip() for line in f]
        if len(self.filenames) == 0:
            assert f"Found 0 files in {self.images_root}"
        print(f"Found {len(self.filenames)} for {subset} set")

        ## Get valid/void class
        file_name = "class-map.csv"
        df = pd.read_csv(f'{root}/{file_name}')
        df = df[['Object/Class Number', 'Object/class Name']]
        self.class_map = df.set_index('Object/Class Number').to_dict()['Object/class Name']
        self.class2id = {cl: i for i,cl in self.class_map.items()}
        valid_cls_name = ['Building', 'Sky', 'Footpath', 'Tree', 'Footpath canopy', 'Car lane', 'Wall',
            'Pedestrian Crossing', 'Pole', 'Small Vehicle', 'Pedestrian']
        self.valid_classes = [self.class2id[cl] for cl in valid_cls_name]
        self.NUM_CLASSES = len(self.valid_classes)
        self.void_classes = [i for i in self.class_map.keys() if i not in self.valid_classes]
        self.train_class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))
        self.ignore_index = ignore_index
        self.input_transform = input_transform

    def encode_segmap(self, label):
        mask = torch.zeros(label.shape[0],label.shape[1], label.shape[2])
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[label == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[label == _validc] = self.train_class_map[_validc]
        return mask
       

    def __getitem__(self, index):
        filename = self.filenames[index]
        image_file = os.path.join(self.images_root, 'Original images', filename)
        ann_file = os.path.join(self.images_root, 'Mask images', filename)
        with open(image_file, 'rb') as f:
            image = Image.open(f).convert('RGB')
            
        with open(ann_file, 'rb') as f:
            label = Image.open(f).convert('P')
        if self.co_transform is not None:
            image, label = self.co_transform(image, label)
        label = self.encode_segmap(label)

        filename = os.path.split(filename)[-1]
        filename = os.path.splitext(filename)[0]
        return image, label, filename

def build_cmap(cmap_dict):
    cmap=np.zeros([len(cmap_dict), 3]).astype(np.uint8)
    for train_id, color in cmap_dict.items():
        color = [int(c.strip()) for c in color.split(",")]
        cmap[train_id,:] = np.array(color)
    return cmap

class footpath(Dataset):

    def __init__(self, root, input_transform=None, subset='val', gt_folder = None):
        self.images_root = os.path.join(root, subset)
        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()
        if len(self.filenames) == 0:
            assert f"Found 0 files in {self.images_root}"

        self.input_transform = input_transform

        ## Get valid/void class
        file_name = "class-map.csv"
        df = pd.read_csv(f'{root}/{file_name}')
        df = df[['Object/Class Number', 'Object/class Name',"RGB Color Code"]]
        self.class_map = df.set_index('Object/Class Number').to_dict()['Object/class Name']
        self.class2id = {cl: i for i,cl in self.class_map.items()}
        valid_cls_name = ['Building', 'Sky', 'Footpath', 'Tree', 'Footpath canopy', 'Car lane', 'Wall',
            'Pedestrian Crossing', 'Pole', 'Small Vehicle', 'Pedestrian']
        self.valid_classes = [self.class2id[cl] for cl in valid_cls_name]
        self.NUM_CLASSES = len(self.valid_classes)
        self.void_classes = [i for i in self.class_map.keys() if i not in self.valid_classes]
        self.train_class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))
        print(self.train_class_map)
        self.ignore_index = len(self.valid_classes)

        # Get color map
        id_cmap = df.set_index('Object/Class Number').to_dict()['RGB Color Code']
        train_id_cmap = {train_id: id_cmap[cls_id] for cls_id, train_id in self.train_class_map.items()}
        train_id_cmap[self.ignore_index] = id_cmap[48]
        print(train_id_cmap)
        self.color_map = build_cmap(train_id_cmap)

        ### TODO clean call gt
        self.gt_folder = gt_folder


    def encode_segmap(self, label):
        mask = torch.zeros(label.shape[0],label.shape[1], label.shape[2])
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[label == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[label == _validc] = self.train_class_map[_validc]
        return mask

       

    def __getitem__(self, index):
        filename = self.filenames[index]
       
        with open(filename, 'rb') as f:
            image = Image.open(f).convert('RGB')
        shape = image.size
        if self.input_transform is not None:
            image = self.input_transform(image)
        filename = os.path.split(filename)[-1]
        filename = os.path.splitext(filename)[0]

        ### GT
        if self.gt_folder is not None:
            with open(f"{self.gt_folder}/{filename}.png", 'rb') as f:
                gt = Image.open(f).convert('P')
            gt = torch.from_numpy(np.array(gt)).long().unsqueeze(0)
            gt = self.encode_segmap(gt)
            return image, gt, filename, shape
        else:
            return image, filename, shape

    def __len__(self):
        return len(self.filenames)

