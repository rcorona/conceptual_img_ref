# Python-level imports.
from torch.utils.data import Dataset
import random
import numpy as np
import os
from PIL import Image


class SUNDataset(Dataset):

    def __init__(self, parent_dir, dtype, img_transform, val_percent=None):

        self._parent_dir = parent_dir
        self._dtype = dtype
        self._img_transform = img_transform

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
        # return 10
        return len(self._data[self._dtype])

    def __getitem__(self, idx):

        row = self._data[self._dtype][idx]

        # Unpack row.
        img_path, class_label = row

        # Process image.
        img = Image.open(os.path.join(self._parent_dir,
                                      img_path)).convert('RGB')
        img = self._img_transform(img)

        # Cast as necessary.
        class_label = int(class_label)

        # Format path for image reference game.
        img_path = '/'.join(img_path.split('/')[1:])

        return {'img': img, 'class_label': class_label, 'img_path': img_path}

    def build_dataset(self):
        # Read in image data and form dataset.
        dataset_ids = {'1': 'train', '0': 'test'}
        self._data = {'train': [], 'test': []}
        with open(os.path.join(self._parent_dir,
                               'train_test_classification_split.txt'), 'r') as f:
            for line in f:
                # Image data line
                # 0: image_id
                # 1: image_path
                # 2: class_id
                # 3: train_test_id
                ls = line.strip().split()
                self._data[dataset_ids[ls[3]]].append([ls[1], int(ls[2]) - 1])

        # Read in class names.
        class_names = [name.split() for name in
                       open(os.path.join(self._parent_dir,
                                         'classes.txt'), 'r')]
        self._class_names = {int(c_id): name for c_id, name in class_names}

        # Read in attribute matrix.
        self._attribute_mat = np.loadtxt(os.path.join(self._parent_dir,
                                                      'attributes_continuous.npy'))

    def attribute_mat(self):
        return self._attribute_mat
