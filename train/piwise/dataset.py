import numpy as np
import os
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
import torch

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)
    #return filename.endswith(".png")

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_path_city(root, name):
    return os.path.join(root, f'{name}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class VOC12(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)


class footpath(Dataset):

    def __init__(self, root, co_transform=None, split=4, subset='val', ignore_index=-1):

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
        self.trainid2class = {train_id: self.class_map[cls_id] for cls_id, train_id in self.train_class_map.items()}

        self.ignore_index = ignore_index
        self.co_transform = co_transform

    def encode_segmap(self, label):
        mask = torch.zeros(label.shape[0],label.shape[1], label.shape[2]).long()
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
            image = load_image(f).convert('RGB')
            
        with open(ann_file, 'rb') as f:
            label = load_image(f).convert('P')
        if self.co_transform is not None:
            image, label = self.co_transform(image, label)
        label = self.encode_segmap(label)

        filename = os.path.split(filename)[-1]
        filename = os.path.splitext(filename)[0]
        return image, label, filename

    def __len__(self):
        return len(self.filenames)


class cityscapes(Dataset):
    def __init__(self, root, co_transform=None, subset='train'):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')
        
        self.images_root += subset
        self.labels_root += subset

        print (self.images_root)
        #self.filenames = [image_basename(f) for f in os.listdir(self.images_root) if is_image(f)]
        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()

        #[os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(".")) for f in fn]
        #self.filenamesGt = [image_basename(f) for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        self.filenamesGt.sort()

        self.co_transform = co_transform # ADDED THIS

    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]
        # image_path_city(self.images_root, filename)
        with open(filename, 'rb') as f:
            image = load_image(f).convert('RGB')
            
        with open(filenameGt, 'rb') as f:
            label = load_image(f).convert('P')

        if self.co_transform is not None:
            image, label = self.co_transform(image, label)

        filename = os.path.split(filename)[-1]
        filename = os.path.splitext(filename)[0]
        return image, label, filename

    def __len__(self):
        return len(self.filenames)

