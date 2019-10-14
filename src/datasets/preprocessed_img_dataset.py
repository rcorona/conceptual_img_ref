import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import sys

class PreprocessedDataset(Dataset):

    def __init__(self, img_dataset, dtype='train', val_percent=None):
        """
        """
        self._data = pickle.load(open(img_dataset, 'rb'))
        self._dtype = dtype

        # Sort based on image path names in case we use with other datasets.
        if 'paths' in self._data[dtype]: 
            sorted_idx = np.argsort(self._data[dtype]['paths'])

            self._data[dtype]['paths'] = [self._data[dtype]['paths'][i] for i in sorted_idx]
            self._data[dtype]['features'] = self._data[dtype]['features'][sorted_idx]
            self._data[dtype]['labels'] = self._data[dtype]['labels'][sorted_idx]

    def vectorize_data(self, img_dataset):
        for dtype in ('train', 'test'):

            features = np.concatenate([np.expand_dims(t[0], 0) for t in self._data[dtype]])
            labels = np.concatenate([np.expand_dims(t[1], 0) for t in self._data[dtype]])

            self._data[dtype] = {'features': features, 'labels': labels}

        # Save vectorized dataset.
        self._data['vectorized'] = None
        pickle.dump(self._data, open(img_dataset, 'wb'))
            
    def normalize_data(self, img_dataset):
        for dtype in ('train', 'test'):

            data = np.concatenate([np.expand_dims(t[0],0) for t in self._data[dtype]])

            # Normalize into [0,1]
            mins = np.tile(np.expand_dims(np.amin(data, 0), 0), (data.shape[0], 1))
            maxes = np.tile(np.expand_dims(np.amax(data, 0), 0), (data.shape[0], 1))

            data = np.divide(data - mins, maxes - mins)

            # Store back into dataset.
            for i in range(data.shape[0]):
                self._data[dtype][i] = (data[i], self._data[dtype][i][1])

        # Save normalized dataset.
        self._data['normalized'] = None
        pickle.dump(self._data, open(img_dataset, 'wb'))
            
    def set_dtype(self, dtype):
        self._dtype = dtype

    def __len__(self):
        return self._data[self._dtype]['features'].shape[0]

    def __getitem__(self, idx):

        # Ignore idx and get random images.
        imgs = self._data[self._dtype]['features'][idx]
        
        return imgs

    def get_paths(self, idx): 

        paths = np.asarray(self._data[self._dtype]['paths'], dtype=object)
        paths = paths[idx]

        return paths

    def delete_imgs(self, delete_list):

        dtype = self._dtype
        indexes = [i for i in range(len(self)) if self._data[dtype]['paths'][i] in delete_list]

        self._data[dtype]['features'] = np.delete(self._data[dtype]['features'], indexes, 0)
        self._data[dtype]['labels'] = np.delete(self._data[dtype]['labels'], indexes)
        self._data[dtype]['paths'] = np.delete(self._data[dtype]['paths'], indexes)
