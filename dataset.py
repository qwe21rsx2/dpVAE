import random
import os
import os.path as osp
import sys
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import datasets as sklsets

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, SVHN

try:
    import cv2
except ImportError:
    cv2 = None

"""
config['data_samples'] = 2000
config['noise'] = 0.05
"""


def get_dataset(config, train=True):
    if config['dataset'].lower() in ['moons', 'swiss_roll']:
        samps = config['data_samples'] if train else config['data_samples']//10
        dataset = GeneratedSet(
          config['dset_dir'], config['dataset'], train=train,
          n_samples=samps, noise=config['data_noise']
        )
    elif config['dataset'].lower() in ['csv', 'table']:
        dataset = CsvDataset(
          config['dset_dir'], train=train,
          feature_cols=config.get('feature_cols'),
          exclude_cols=config.get('exclude_cols'),
          categorical_cols=config.get('categorical_cols'),
          normalize=config.get('normalize', True),
          val_ratio=config.get('val_ratio', 0.2),
          seed=config.get('seed', 1)
        )
    elif config['dataset'].lower() in ['mnist', 'svhn', 'dsprite']:
        dataset = PreDefSet( config['dset_dir'], type=config['dataset'].lower(),
          train=train, download=True, augment=config['augment']&train
        )

    elif config['dataset'].lower() in ['celeba']:
        img_size=config['image_size'] if 'image_size' in config else 32
        dataset = FolderDataset(
          config['dset_dir'], img_size=img_size, train=train, augment=config['augment']&train
        )
        # if not train:
        #     indx = torch.randint(len(dataset), (10000,))
        #     dataset.data = [dataset.data[idx] for idx in indx]

    if train and config['type'] == 'factor':
        dataset = FactorSet(dataset)


    return DataLoader(
      dataset,
      batch_size=config['batch_size'],
      shuffle=train,
      num_workers=config['num_workers'],
      pin_memory=True,
      drop_last=True
    )


class FactorSet(Dataset):
    """docstring for FactorSet."""

    def __init__(self, dataset):
        super(FactorSet, self).__init__()
        self.data = dataset
        self.indices = range(len(self))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_1 = self.data[idx]
        idx2 = random.choice(self.indices)
        x_2 = self.data[idx2]
        return x_1, x_2


class GeneratedSet(Dataset):
    """docstring for GeneratedSet."""

    def __init__(self, location=None, setype='moons', train=False, n_samples=2000, noise=0.05):
        super(GeneratedSet, self).__init__()

        generate = False
        if location is None:
            generate = True
            location = 'gen_data/'
        os.makedirs(location, exist_ok=True)
        dfile = (setype + '.npy') if train else (setype + '_val.npy')
        dfile = osp.join(location, dfile)
        if osp.exists(dfile):
            self.data = np.load(dfile)
        else:
            generate = True
        if generate:
            if setype == 'moons':
                self.data = sklsets.make_moons(
                    n_samples=n_samples, noise=noise
                )[0].astype(np.float32)
            elif setype == 'swiss_roll':
                self.data = sklsets.make_swiss_roll(
                    n_samples=n_samples, noise=noise
                )[0].astype(np.float32)
            np.save(dfile, self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])


class CsvDataset(Dataset):
    """Numeric CSV dataset for linear VAE models."""

    def __init__(self, csv_path, train=True, feature_cols=None, exclude_cols=None,
                 categorical_cols=None, normalize=True, val_ratio=0.2, seed=1):
        super(CsvDataset, self).__init__()
        categorical_cols = set(categorical_cols or [])
        exclude_cols = set(exclude_cols or [])

        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            columns = reader.fieldnames

        if not rows:
            raise ValueError('CSV file is empty: {}'.format(csv_path))

        if feature_cols is None:
            feature_cols = [col for col in columns if col not in exclude_cols]

        encoders = {}
        data = []
        for row in rows:
            values = []
            for col in feature_cols:
                value = row[col]
                if value == '':
                    values.append(np.nan)
                elif col in categorical_cols:
                    mapping = encoders.setdefault(col, {})
                    if value not in mapping:
                        mapping[value] = len(mapping)
                    values.append(float(mapping[value]))
                else:
                    values.append(float(value))
            data.append(values)

        data = np.asarray(data, dtype=np.float32)
        means = np.nanmean(data, axis=0)
        inds = np.where(np.isnan(data))
        data[inds] = np.take(means, inds[1])

        rng = np.random.RandomState(seed)
        indices = np.arange(len(data))
        rng.shuffle(indices)
        split = int(len(indices) * (1 - val_ratio))
        train_idx = indices[:split]
        val_idx = indices[split:]

        if normalize:
            mean = data[train_idx].mean(axis=0)
            std = data[train_idx].std(axis=0)
            std[std == 0] = 1
            data = (data - mean) / std

        self.data = data[train_idx if train else val_idx]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])


class PreDefSet(Dataset):
    """docstring for PreDefSet."""

    def __init__(self, root, type='mnist', train=True, augment=False, download=False):
        super(PreDefSet, self).__init__()
        dotrans = []
        if augment:
            dotrans += [
                transforms.RandomApply([transforms.ColorJitter(
                    0.5,0.5,0.5,0.25)], 0.5),
                transforms.RandomApply([transforms.RandomAffine(
                    5, translate=(0.1,0.1), scale=(0.9,1.1))], 0.5)
            ]
        dotrans += [
            transforms.ToTensor(),
            # transforms.Lambda(expand3chans)
        ]
        dotrans = transforms.Compose(dotrans)
        if type == 'mnist':
            self.data = MNIST(root,
              train=train, download=download, transform=dotrans
            )
            if train:
                np.save(root + 'trainLabels.npy', self.data.train_labels.numpy())
            else:
                np.save(root + 'valLabels.npy', self.data.train_labels.numpy())
        elif type == 'svhn':
            self.data = SVHN(root,
              split='train' if train else 'test',
              download=download, transform=dotrans
            )


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0]


class FolderDataset(Dataset):
    def __init__(self, folder_path, img_size, train=None, augment=False):
        if train is not None:
            folder_path = osp.join(folder_path, 'CelebA_Cropped_Train') if train else osp.join(folder_path, 'CelebA_Cropped_Val')
        self.data = [osp.join(folder_path, filep) for filep in os.listdir(folder_path)]

        self.img_size = (img_size, img_size)
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if cv2 is None:
            raise ImportError('opencv-python is required for FolderDataset image resizing')
        img_path = self.data[index % len(self.data)]
        # img = cv2.imread(img_path, 1)
        img = np.array(Image.open(img_path))
        # img = plt.imread(img_path)
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_CUBIC)
        img = transforms.ToTensor()(img)
        return img



def main():
    dataset = FolderDataset('/home/sci/riddhishb/Downloads/DatasetsCVPR/img_align_celeba/', img_size=64)
    sample = dataset[0]
    print(sample.shape)
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
