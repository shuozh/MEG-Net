import torch
import numpy as np
import torch.utils.data as data
from func_input_npy import image_input_full
import os


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        self.output_size = (output_size, output_size)

    def __call__(self, train_data, gt_image, scale):
        h, w = train_data.shape[2], train_data.shape[3]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        train_data_tmp = train_data[:, :, top: top + new_h, left: left + new_w]

        # gtdata_tmp = gt_image[:, :, top * scale: top * scale + new_h * scale -scale+1,
        #              left * scale: left * scale + new_w * scale -scale+1] ？？？？？？
        gtdata_tmp = gt_image[:, :, top * scale: top * scale + new_h * scale,
                     left * scale: left * scale + new_w * scale]

        return train_data_tmp, gtdata_tmp


class SRLF_Dataset_new(data.Dataset):
    """Light Field dataset."""

    def __init__(self, dir_LF, repeat_size=32, view_n=9, scale=2, crop_size=32, if_flip=False, if_rotation=False,
                 if_test=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.crop_size = crop_size
        self.repeat_size = repeat_size
        self.view_n = view_n
        self.RandomCrop = RandomCrop(crop_size)
        self.if_flip = if_flip
        self.if_rotation = if_rotation
        self.if_test = if_test
        self.scale = scale
        self.dir_LF = dir_LF
        self.train_data_all = []
        self.gt_data_all = []
        self.numbers = len(os.listdir(dir_LF))

        img_list = os.listdir(dir_LF)
        img_list.sort()
        for img_name in img_list:
            train_data, gt_data = image_input_full(dir_LF, img_name, view_n, scale)
            self.train_data_all.append(train_data)
            self.gt_data_all.append(gt_data)

    def __len__(self):
        return self.repeat_size * self.numbers

    def __getitem__(self, idx):

        train_data = self.train_data_all[idx // self.repeat_size]
        gt_data = self.gt_data_all[idx // self.repeat_size]

        if self.if_test:
            return torch.from_numpy(train_data.copy()), torch.from_numpy(gt_data.copy())

        else:
            train_data, gt_data = self.RandomCrop(train_data, gt_data, self.scale)

            if self.if_flip:
                random_tmp = np.random.random()
                if random_tmp >= (2.0 / 3):

                    train_data = np.flip(train_data, 2)
                    train_data = np.flip(train_data, 0)

                    gt_data = np.flip(gt_data, 2)
                    gt_data = np.flip(gt_data, 0)

                elif random_tmp <= (1.0 / 3):

                    train_data = np.flip(train_data, 3)
                    train_data = np.flip(train_data, 1)

                    gt_data = np.flip(gt_data, 3)
                    gt_data = np.flip(gt_data, 1)

            if self.if_rotation:
                random_tmp = np.random.random()
                if random_tmp >= (1.0 / 2):
                    train_data = np.rot90(train_data, 2, (2, 3))
                    train_data = np.flip(train_data, 0)
                    train_data = np.flip(train_data, 1)

                    gt_data = np.rot90(gt_data, 2, (2, 3))
                    gt_data = np.flip(gt_data, 0)
                    gt_data = np.flip(gt_data, 1)

            return torch.from_numpy(train_data.copy()), torch.from_numpy(gt_data.copy())
