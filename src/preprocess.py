"""

"""
import os
import torch
from torch.utils.data import Dataset  # NOQA
import torchvision.transforms.functional as tf
import numpy as np
import cv2 as cv
import pandas as pd
from tqdm import tqdm
import random
import matplotlib.pyplot as plt


class VideoDataset(Dataset):
    def __init__(self, path_frames, idx_frames, path_labels, input_shape,
                 num_samples=None, data_augmentation=False):
        """

        """
        self.path_frames = path_frames
        self.idx_frames = idx_frames
        self.labels = np.load(path_labels)
        self.input_shape = input_shape
        self.num_samples = num_samples
        self.data_augmentation = data_augmentation

    def __len__(self):
        """

        """
        if self.num_samples is None:
            return len(self.idx_frames)
        else:
            return self.num_samples

    def __getitem__(self, idx):
        """

        """
        imgs = []
        idx_last_frame = self.idx_frames[idx]
        # loop through the input frames
        for idx_actual_frame in range(idx_last_frame - (self.input_shape[0] - 1) * 1, idx_last_frame + 1, 1):
            # read in image and basic transformations
            img_actual = cv.imread(os.path.join(self.path_frames, "{:06d}".format(idx_actual_frame) + ".jpg"))
            # to grayscale
            if self.input_shape[1] == 1:
                img_actual = cv.cvtColor(img_actual, cv.COLOR_BGR2GRAY)
                img_actual = img_actual[:, :, np.newaxis]
            # center croping
            img_actual = self.center_crop(img_actual, self.input_shape)
            # normalize
            img_actual = img_actual/256
            # append to list
            imgs.append(img_actual)

        # augmentate image
        if self.data_augmentation:
            imgs = self.img_transforms(imgs)
        # convert list of images to tensor
        imgs = np.swapaxes(np.asarray(imgs, dtype=np.float32), 1, 3)
        # return outputs
        return imgs, self.labels[idx_last_frame]

    @staticmethod
    def center_crop(img, output_shape):
        """

        """
        pix_vertical_crop = (img.shape[1] - output_shape[3]) // 2
        pix_horizontal_crop = (img.shape[0] - output_shape[2]) // 2
        img = img[pix_horizontal_crop:-pix_horizontal_crop, pix_vertical_crop:-pix_vertical_crop]
        return img

    @staticmethod
    def img_transforms(imgs):
        """

        """
        # transformation parameters
        # parameters for one channel images
        brightness_factor = random.uniform(0.25, 1.75)
        gamma = random.uniform(0.25, 1.75)
        vflip = random.uniform(0, 1) > 0.5
        # additional parameters for three channels images
        contrast_factor = random.uniform(0.75, 1.25)
        hue_factor = random.uniform(-0.25, 0.25)
        saturation_factor = random.uniform(0, 0)

        # loop through all input images
        if isinstance(imgs, list) is False:
            imgs = list(imgs)
        imgs_output = []
        for img_actual in imgs:
            # to tensor with shape (C x H x W)
            img_actual = torch.from_numpy(np.swapaxes(img_actual, 0, 2))

            # transformations
            img_actual = tf.adjust_brightness(img_actual, brightness_factor=brightness_factor)
            img_actual = tf.adjust_gamma(img_actual, gamma=gamma)
            if vflip:
                img_actual = tf.vflip(img_actual)
            if img_actual.shape[0] == 3:
                img_actual = tf.adjust_contrast(img_actual, contrast_factor=contrast_factor)
            if img_actual.shape[0] == 3:
                img_actual = tf.adjust_hue(img_actual, hue_factor=hue_factor)
            if img_actual.shape[0] == 3:
                img_actual = tf.adjust_saturation(img_actual, saturation_factor=saturation_factor)

            # back to numpy array (H x W x C) and append to list
            img_actual = img_actual.numpy()
            img_actual = np.swapaxes(img_actual, 0, 2)
            """
            plt.imshow(img_actual, vmin=0, vmax=1, cmap="gray")
            plt.show()
            """
            imgs_output.append(img_actual)

        return imgs_output

    @staticmethod
    def save_video_to_frames(path_video, path_labels, path_save_frames, path_save_labels):
        """

        """
        # loop through the video
        video_capture = cv.VideoCapture(path_video)
        success, frame = video_capture.read()
        num_total_frames = int(video_capture.get(cv.CAP_PROP_FRAME_COUNT))
        for idx_img in tqdm(range(num_total_frames)):
            cv.imwrite(os.path.join(path_save_frames, "{:06d}".format(idx_img) + ".jpg"), frame)
            success, frame = video_capture.read()

        # save label as numpy arrays
        np.save(path_save_labels, pd.read_csv(path_labels, header=None).to_numpy(dtype=np.float32))

    @staticmethod
    def split_train_val_set(path_frames, num_data_splits, num_samples,
                            num_input_frames, train_val_split, shuffle):
        """

        """
        # if num_samples is specified, all images in the given path are taken
        if num_samples is None:
            num_samples = len(os.listdir(path_frames))

        # loop through the data
        img_names_train_set = []
        img_names_test_set = []
        for idx_img in range(num_input_frames - 1, num_samples):
            # shuffle split
            if shuffle:
                if random.uniform(0, 1) < train_val_split:
                    img_names_train_set.append(idx_img)
                else:
                    img_names_test_set.append(idx_img)

            # non shuffle split
            else:
                num_samples_pro_split = num_samples // num_data_splits
                split_actual = idx_img // num_samples_pro_split
                idx_img_normalized = idx_img - split_actual * num_samples_pro_split
                if idx_img_normalized < num_samples_pro_split * train_val_split:
                    img_names_train_set.append(idx_img)
                else:
                    img_names_test_set.append(idx_img)

        # return the data split
        return img_names_train_set, img_names_test_set

    @staticmethod
    def plot_labels_histogram(path_labels, idx_img_train_set, idx_img_test_set):
        """

        """
        labels = np.load(path_labels)
        label_train = labels[idx_img_train_set]
        labels_test = labels[idx_img_test_set]
        fig, axs = plt.subplots()
        axs.hist(label_train, bins=150, label="train")
        axs.hist(labels_test, bins=150, label="test")
        axs.set_title("Histogram dataset")
        plt.legend()
        plt.grid()
