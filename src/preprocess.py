"""

"""
import os
from torch.utils.data import Dataset  # NOQA
import cv2 as cv
import pandas as pd
import numpy as np
from tqdm import tqdm
import random


class VideoDataset(Dataset):
    def __init__(self, path_frames, idx_frames, path_labels, input_shape,
                 num_samples=None, normalize_ouput=False):
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
        for idx_actual_frame in range(idx_last_frame - self.input_shape[0], idx_last_frame):
            # read in image
            img = cv.imread(os.path.join(self.path_frames, "{:06d}".format(idx_actual_frame) + ".jpg"))
            # noramlize and append to list
            list_img.append(self.normalize_img(img, (self.input_shape[2], self.input_shape[3], self.input_shape[1])))

        x = np.swapaxes(np.array(list_img, dtype=np.float32), 1, 3)
        y = self.labels[idx_actual_frame]  # NOQA
        return x, y

    @staticmethod
    def normalize_img(img, shape):
        """

        """
        # reshape image
        img = cv.resize(img, (shape[0], shape[1]))
        if shape[2] == 1:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = img[:, :, np.newaxis]
        # normalize image
        img = img / 256
        # return image
        return img

    @staticmethod
    def denormalize_img(img, shape):
        """

        """
        pass

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
    def split_train_test_set(path_frames, train_val_split, num_frames):
        """

        """
        img_names_train_set = []
        img_names_test_set = []
        num_samples = len(os.listdir(path_frames))
        for idx_img in range(num_frames, num_samples):
            if random.uniform(0, 1) < train_val_split:
                img_names_train_set.append(idx_img)
            else:
                img_names_test_set.append(idx_img)

        return img_names_train_set, img_names_test_set
