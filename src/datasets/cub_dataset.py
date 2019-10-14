import torch
from torch.utils.data import Dataset
import random
import numpy as np
import pickle
import os
from PIL import Image
import sys

class CUBDataset(Dataset):

    def __init__(self, parent_dir, dtype, img_transform, val_percent=None):

        self._parent_dir = parent_dir
        self._dtype = dtype            
        self._img_transform = img_transform
        self._attr_types = None
        
        # Load data. 
        self.build_dataset()

        # Generate validation set if desired.
        if val_percent:
            self.gen_fold(val_percent)

    def set_dtype(self, dtype):
        self._dtype = dtype
            
    def gen_fold(self, val_percent):

        # Generate random training/validation split. 
        n_val = int(val_percent * len(self._data['train']))
        random.shuffle(self._data['train'])

        self._data['val'] = self._data['train'][:n_val]
        self._data['train'] = self._data['train'][n_val:]                

    def __len__(self):
        return len(self._data[self._dtype])

    def __getitem__(self, idx):

        row = self._data[self._dtype][idx]

        # Unpack row.
        img_path, class_label = row

        # Process image.
        img = Image.open(os.path.join(self._parent_dir, 'images/' + img_path)).convert('RGB')
        img = self._img_transform(img)

        # Cast as necessary.
        class_label = int(class_label)

        return {'img': img, 'class_label': class_label, 'img_path': img_path}

    def build_dataset(self):

        # Read in image data split ID. 
        split_ids = [line.strip().split() for line in \
                           open(os.path.join(self._parent_dir, 'train_test_split.txt'), 'r')]
        split_ids = {l[0]: l[1] for l in split_ids}
        
        # Read in image paths.
        image_paths = [line.strip().split() for line in \
                       open(os.path.join(self._parent_dir, 'images.txt'), 'r')]
        image_paths = {l[0]: l[1] for l in image_paths}

        # Read in class names.
        class_names = [name.split() for name in \
                       open(os.path.join(self._parent_dir, 'classes.txt'), 'r')]
        self._class_names = {int(c_id) - 1: name for c_id, name in class_names}

        # Read in attribute matrix.
        attribute_dir = os.path.join(self._parent_dir, 'attributes')
        attribute_path = os.path.join(attribute_dir, 'class_attribute_labels_continuous.txt')
        attribute_file = [line.split() for line in open(attribute_path, 'r')]

        self._attribute_mat = np.zeros((312, 200))

        for c in range(200):
            for r in range(312):
                self._attribute_mat[r,c] = float(attribute_file[c][r]) / 100.0

        # Read in attribute names.
        name_file_path = os.path.join(attribute_dir, 'attributes.txt')
        name_file = [line.strip().split() for line in open(name_file_path, 'r')]
        self._attribute_names = {int(a[0]) - 1: a[1] for a in name_file}
        
        # Form dataset.
        dataset_ids = {'1': 'train', '0': 'test'} 
        self._data = {'train': [], 'test': []}

        # Read images into tables.
        for image_id in image_paths:

            # Extract information. 
            data_id = dataset_ids[split_ids[image_id]]
            img_path = image_paths[image_id]
            img_class = int(img_path.split('.')[0]) - 1

            # Place in pertinent set.
            self._data[data_id].append([img_path, img_class])

    def attribute_mat(self):
        return self._attribute_mat
