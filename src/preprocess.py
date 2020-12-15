"""

"""
import os
from torch.utils.data import Dataset  # NOQA
import cv2 as cv
import pandas as pd
import numpy as np
from tqdm import tqdm


class VideoDataset(Dataset):
    def __init__(self, path_save_img, path_label_speed, input_shape, num_samples=None, normalize_ouput=False):
        """

        """
        self.path_save_img = path_save_img
        self.speed = np.load(path_label_speed)
        self.speed_max = max(self.speed)[0]
        self.speed_min = min(self.speed)[0]
        if normalize_ouput:
            self.speed = self.normalize_speed(self.speed, self.speed_max, self.speed_min)
        self.input_shape = input_shape
        self.num_samples = num_samples

    def __len__(self):
        """

        """
        if self.num_samples is None:
            return len(os.listdir(self.path_save_img)) - self.input_shape[0] + 1
        else:
            return self.num_samples

    def __getitem__(self, idx):
        """

        """
        list_img = []
        for img_counter in range(idx, idx + self.input_shape[0]):
            # read in image
            img = cv.imread(os.path.join(self.path_save_img, "{:06d}".format(img_counter) + ".jpg"))
            # noramlize and append to list
            list_img.append(self.normalize_img(img,
                                               (self.input_shape[2], self.input_shape[3], self.input_shape[1])))

        x = np.swapaxes(np.array(list_img, dtype=np.float32), 1, 3)
        y = self.speed[img_counter]  # NOQA
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
    def save_video_to_frames(path_video, path_label, path_save_img, train_val_split):
        """

        """
        video_capture = cv.VideoCapture(path_video)
        success, frame = video_capture.read()
        num_total_frames = int(video_capture.get(cv.CAP_PROP_FRAME_COUNT))

        label = pd.read_csv(path_label, header=None).to_numpy(dtype=np.float32)
        label_train_set = []
        label_test_set = []

        num_total_frames_train = 0
        num_total_frames_val = 0
        for img_counter in tqdm(range(num_total_frames)):
            # train set
            if img_counter < num_total_frames * train_val_split:
                cv.imwrite(os.path.join(path_save_img, "train_set",
                                        "{:06d}".format(num_total_frames_train) + ".jpg"), frame)
                label_train_set.append(label[img_counter])
                num_total_frames_train += 1
            # test set
            else:
                cv.imwrite(os.path.join(path_save_img, "test_set",
                                        "{:06d}".format(num_total_frames_val) + ".jpg"), frame)
                label_test_set.append(label[img_counter])
                num_total_frames_val += 1
            success, frame = video_capture.read()

        # save label as numpy arrays
        np.save(os.path.join(path_save_img, "train_set", "train_label.npy"), np.asarray(label_train_set))
        np.save(os.path.join(path_save_img, "test_set", "test_label.npy"), np.asarray(label_test_set))


def low_pass_filter(x, y_init=None, fps=30, time_constant=1):
    """

    """
    # parameters of the low pass filter
    lowpass_cutoff = 1 / time_constant
    dt = 1 / fps
    rc = 1 / (2 * np.pi * lowpass_cutoff)
    alpha = dt / (rc + dt)
    # create output array
    y = np.zeros_like(x)
    if y_init is None:
        y[0] = x[0]
    else:
        y[0] = y_init
    for idx in range(1, x.shape[0]):
        y[idx] = x[idx] * alpha + (1 - alpha) * y[idx - 1]
    # return output
    return y
