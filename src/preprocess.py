"""

"""
import os
from torch.utils.data import Dataset  # NOQA
import cv2 as cv
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

STRIDE_MAX = 2


class VideoDataset(Dataset):
    def __init__(self, path_frames, idx_frames, path_labels, input_shape,
                 num_samples=None, normalize_ouput=False, data_augmentation=False, stride=None):
        """

        """
        self.path_frames = path_frames
        self.idx_frames = idx_frames
        self.labels = np.load(path_labels)
        self.label_max = max(self.labels)[0]
        self.label_min = min(self.labels)[0]
        if normalize_ouput:
            self.labels = self.normalize_speed(self.labels, self.label_max, self.label_min)
        self.input_shape = input_shape
        self.num_samples = num_samples
        self.data_augmentation = data_augmentation
        self.stride = stride

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
        list_img = []
        idx_last_frame = self.idx_frames[idx]
        if self.stride is None:
            stride = random.randrange(1, min((idx_last_frame + 1) // self.input_shape[0] + 1, STRIDE_MAX + 1))
        else:
            stride = self.stride
        # loop through the input frames
        for idx_actual_frame in range(idx_last_frame - (self.input_shape[0] - 1)*stride, idx_last_frame + 1, stride):
            assert idx_actual_frame >= 0
            # read in image
            img = cv.imread(os.path.join(self.path_frames, "{:06d}".format(idx_actual_frame) + ".jpg"))
            # normalize and append to list
            list_img.append(img)

        # augmentate image
        if self.data_augmentation:
            flip = random.uniform(0, 1) > 0.5
            gamma = random.uniform(0.75, 1.5)
        else:
            flip = False
            gamma = 1
        x = self.preprocess_img(list_img,
                                shape=(self.input_shape[2], self.input_shape[3], self.input_shape[1]),
                                flip=flip,
                                gamma=gamma)
        x = np.swapaxes(x, 1, 3)

        # pick out label
        assert idx_last_frame == idx_actual_frame # NOQA
        y = self.labels[idx_last_frame]*stride

        # return output
        return x, y

    @staticmethod
    def preprocess_img(list_img, shape, flip=False, gamma=1):
        """

        """
        output = np.zeros((len(list_img), shape[0], shape[1], shape[2]), dtype=np.float32)
        for idx, img in enumerate(list_img):
            # center crop:
            img = img[128:-128, 208:-208]
            # reshape image if needed
            if img.shape != shape:
                img = cv.resize(img, (shape[0], shape[1]))
            # transform to gray scale if needed
            if shape[2] == 1:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                img = img[:, :, np.newaxis]
            # flip image if needed
            if flip:
                img = np.fliplr(img)
            # adjust gamma if needed
            img = VideoDataset.adjust_gamma(img, gamma)
            # normalize image
            img = img / 256
            # append image
            output[idx, :, :, :] = img

        # return image
        return output

    @staticmethod
    def adjust_gamma(img, gamma=1.0):
        """

        """
        gamma_inv = 1.0 / gamma
        table = np.array([((i / 255.0) ** gamma_inv) * 255 for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv.LUT(img, table)

    @staticmethod
    def normalize_speed(speed, speed_max, speed_min):
        """

        """
        return (speed - speed_min) / (speed_max - speed_min)

    @staticmethod
    def denormalize_speed(speed, speed_max, speed_min):
        """

        """
        return (speed * (speed_max - speed_min)) + speed_min

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
    def split_train_test_set(path_frames, data_splits, num_samples, num_input_frames, train_val_split, shuffle):
        """

        """
        # if num_samples is specified, then no all the samples are taking
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
                num_samples_pro_split = num_samples // data_splits
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
        plt.hist(label_train, bins=30)
        plt.hist(labels_test, bins=30)
        plt.show()
