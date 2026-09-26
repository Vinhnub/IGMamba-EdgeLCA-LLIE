import os
import random
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from data.util import is_image_file, load_img
import torch.nn.functional as F

class SmallDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(SmallDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform

        folder = join(self.data_dir, 'low')
        folder2 = join(self.data_dir, 'high')

        self.data_filenames = sorted([join(folder, x) for x in listdir(folder) if is_image_file(x)])
        self.data_filenames2 = sorted([join(folder2, x) for x in listdir(folder2) if is_image_file(x)])
        assert len(self.data_filenames) == len(self.data_filenames2), (
            f"Mismatch in image counts: {len(self.data_filenames)} in low vs {len(self.data_filenames2)} in high"
        )

    def __getitem__(self, index):
        im1 = load_img(self.data_filenames[index])
        im2 = load_img(self.data_filenames2[index])
        _, file1 = os.path.split(self.data_filenames[index])
        _, file2 = os.path.split(self.data_filenames2[index])

        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed)
        if self.transform:
            random.seed(seed)
            torch.manual_seed(seed)
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)
            im2 = self.transform(im2)
        return im1, im2, file1, file2

    def __len__(self):
        return len(self.data_filenames)


class SmallDatasetFromFolderEval(data.Dataset):
    def __init__(self, data_dir, transform=None, factor=8):
        super(SmallDatasetFromFolderEval, self).__init__()
        data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        data_filenames.sort()
        self.data_filenames = data_filenames
        self.transform = transform
        self.factor = factor

    def __getitem__(self, index):
        input = load_img(self.data_filenames[index])
        _, file = os.path.split(self.data_filenames[index])

        if self.transform:
            input = self.transform(input)
            factor = self.factor
            h, w = input.shape[1], input.shape[2]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            if padh > 0 or padw > 0:
                input = F.pad(input.unsqueeze(0), (0, padw, 0, padh), 'reflect').squeeze(0)
        return input, file, h, w

    def __len__(self):
        return len(self.data_filenames)
