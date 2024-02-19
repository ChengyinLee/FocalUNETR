#%%
import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
# from scipy.ndimage.interpolation import zoom
from scipy.ndimage import zoom
from torch.utils.data import Dataset
# import augmentation
from . import augmentation


def random_rot_flip(image, label, contour):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    if contour is not None:
        contour = np.rot90(contour, k)
    
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    if contour is not None:
        contour = np.flip(contour, axis=axis).copy()
    
    return image, label, contour


def random_rotate(image, label, contour):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    if contour is not None:
        contour = ndimage.rotate(contour, angle, order=0, reshape=False)
    return image, label, contour


class RandomGenerator(object):
    def __init__(self, output_size, rotate_or_flip_prob=0.5):
        self.output_size = output_size
        self.rotate_or_flip_prob = rotate_or_flip_prob

    def __call__(self, sample):
        image, label, contour = sample['image'], sample['label'], sample['contour']
        if random.random() < self.rotate_or_flip_prob:
            image, label, contour = random_rot_flip(image, label, contour)
        elif random.random() < self.rotate_or_flip_prob:
            image, label, contour = random_rotate(image, label, contour)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            if contour is not None:
                contour = zoom(contour, (self.output_size[0] / x, self.output_size[1] / y), order=3)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        if contour is not None:
            contour = torch.from_numpy(contour.astype(np.float32))
        sample = {'image': image, 'label': label.long(), 'contour': contour}
        return sample


class ProstateDataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None, transform_more=True, contour_key = 'contour1p6'):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.transform_more = transform_more
        self.contour_key = contour_key

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        """
        h5py data
        keys:  ['contour00', 'contour0p4', 'contour0p8', 'contour1p2', 'contour1p6', 'contour2p0', 'contour3p0', 'image', 'label']
        contour used for training
        """
        if self.split in ["train"]:
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, 'contours')
            data_path = os.path.join(data_path, slice_name +'.npy.h5')
            data = h5py.File(data_path)
            image, label, contour =  data['image'][:], data['label'][:], data[self.contour_key][:]
        elif self.split in ["val_vol","test_vol"]:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = os.path.join(self.data_dir, 'by_case_h5')
            filepath = filepath + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label, contour = data['image'][:], data['label'][:], 1.0
        else:
            image, label, contour = None, None, None
            assert image is not None
        
        sample = {'image': image, 'label': label, 'contour': contour}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        
        # more augmentation for the train
        if self.split in ["train"] and self.transform_more:
            tensor_img = sample['image'].unsqueeze(0)
            # Gaussian Noise
            tensor_img = augmentation.gaussian_noise(tensor_img, std=0.02)
            # Additive brightness
            tensor_img = augmentation.brightness_additive(tensor_img, std=0.1)
            # gamma
            tensor_img = augmentation.gamma(tensor_img, gamma_range=[0.5, 1.6], retain_stats=True)
            sample['image'] = tensor_img.squeeze(0)
        return sample