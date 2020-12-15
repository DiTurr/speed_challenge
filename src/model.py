"""

"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F  # NOQA
from efficientnet_pytorch import EfficientNet
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.preprocess import VideoDataset


class SpeedChallengeModel:
    def __init__(self, input_shape):
        """

        """
        self.model = EfficientNetConvLSTM(input_shape)
        if torch.cuda.is_available():
            self.model.cuda()
        self.input_shape = input_shape
        self.num_epochs = None
        self.training_generator = None
        self.validation_generator = None
        self.optimizer = None
        self.loss_function = None
        self.history = None

    def train(self, num_epochs, training_generator, validation_generator, optimizer, loss_function):
        """

        """
        # set attributes
        self.num_epochs = num_epochs
        self.training_generator = training_generator
        self.validation_generator = validation_generator
        self.optimizer = optimizer
        self.loss_function = loss_function

        # CUDA for PyTorch
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        # loop over epochs
        self.history = {"loss": [], "val_loss": []}
        for epoch in range(self.num_epochs):
            # create progress bar
            pbar = tqdm(total=len(self.training_generator) + len(self.validation_generator),
                        ncols=100, dynamic_ncols=True,
                        desc=str(epoch + 1).zfill(5) + "/" + str(self.num_epochs).zfill(5) + ": ")
            # train epoch
            mean_loss_epoch_train, mean_loss_epoch_val = self.train_epoch(device, pbar)

            # save history information
            if mean_loss_epoch_train is not None:
                self.history["loss"].append(mean_loss_epoch_train)
            if mean_loss_epoch_val is not None:
                self.history["val_loss"].append(mean_loss_epoch_val)

            # close progress bar
            pbar.close()

        # ensure progress bar is closed
        pbar.close()  # NOQA

    def train_epoch(self, device, pbar):
        """

        """
        # training
        running_loss_batch_train = 0
        mean_loss_batch_train = None
        with torch.set_grad_enabled(True):
            for index, (x_batch, y_batch) in enumerate(self.training_generator):
                # transfer to GPU
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                # set model to training mode
                self.model.train()

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                y_hat_batch = self.model(x_batch)
                loss_batch_train = self.loss_function(y_batch, y_hat_batch)
                loss_batch_train.backward()
                self.optimizer.step()

                # printing information/statistics
                running_loss_batch_train += loss_batch_train.item()
                mean_loss_batch_train = running_loss_batch_train / (index + 1)
                pbar.update(1)
                pbar.set_postfix_str("Loss: {:.4f}".format(mean_loss_batch_train))

        # validation
        running_loss_batch_val = 0
        mean_loss_batch_val = None
        with torch.set_grad_enabled(False):
            for index, (x_batch, y_batch) in enumerate(self.validation_generator):
                # transfer to GPU
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                # set model to evaluate model and do forward computation
                self.model.eval()
                y_hat_batch = self.model(x_batch)
                loss_batch_val = self.loss_function(y_batch, y_hat_batch)

                # printing information/statistics
                running_loss_batch_val += loss_batch_val.item()
                mean_loss_batch_val = running_loss_batch_val / (index + 1)
                pbar.update(1)
                pbar.set_postfix_str("Loss: {:.4f}; Validation Loss: {:.4f}".
                                     format(mean_loss_batch_train, mean_loss_batch_val))

        # return value
        return mean_loss_batch_train, mean_loss_batch_val

    def predict(self, path_save_img, path_label_speed, batch_size, num_samples=None, normalize_ouput=False):
        """

        """
        # preprocess function inputs
        if num_samples is None:
            num_samples = len(os.listdir(path_save_img)) - self.input_shape[0] + 1
        speed = np.load(path_label_speed)

        # CUDA for PyTorch
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        # loop through samples
        y = []
        y_hat = []
        for idx_sample in tqdm(range(0, num_samples, batch_size)):
            # loop through the batches
            list_img = []
            for idx_batch in range(idx_sample, idx_sample + batch_size):
                # loop through input frames
                list_img_tmp = []
                for img_counter in range(idx_batch, idx_batch + self.input_shape[0]):
                    # read in image
                    img = cv.imread(os.path.join(path_save_img, "{:06d}".format(img_counter) + ".jpg"))
                    # noramlize and append to list
                    img = VideoDataset.normalize_img(img, (self.input_shape[2],
                                                           self.input_shape[3],
                                                           self.input_shape[1]))
                    # append image to temporal image list
                    list_img_tmp.append(img)

                # append to batch image list
                list_img.append(list_img_tmp)
                y.append(speed[img_counter])  # NOQA

            # convert to numpy array and move to GPU
            x = np.swapaxes(np.array(list_img, dtype=np.float32), 2, -1)
            x = torch.from_numpy(x)
            x = x.to(device)

            # set model to evaluate model and do forward computation
            self.model.eval()
            prediction = self.model(x)
            prediction = prediction.cpu().detach().numpy().reshape((-1, 1))
            if normalize_ouput:
                prediction = VideoDataset.denormalize_speed(prediction,
                                                            self.training_generator.dataset.speed_max,
                                                            self.training_generator.dataset.speed_min)

            # append results
            y_hat.append(prediction)

        # return
        y = np.array(y).reshape((-1, 1))
        y_hat = np.array(y_hat).reshape((-1, 1))
        return y, y_hat

    def plot_history(self):
        """

        """
        if self.history is not None:
            fig, axs = plt.subplots()
            axs.plot(self.history["loss"], label="loss")
            axs.plot(self.history["val_loss"], label="val_loss")
            axs.set_title("Losses")
            plt.legend()
            plt.grid()
        else:
            print("[ERROR] no history to plot ...")

    @staticmethod
    def plot_prediction(y, y_hat, labels):
        fig, axs = plt.subplots()
        axs.plot(y, label="y")
        for idx, y_hat_actual in enumerate(y_hat):
            axs.plot(y_hat_actual, label=labels[idx])
        axs.set_title("Speed Prediction")
        plt.legend()
        plt.grid()

    @staticmethod
    def plot_show():
        """

        """
        plt.show()


class EfficientNetConvLSTM(nn.Module):
    def __init__(self, shape):
        """

        """
        super(EfficientNetConvLSTM, self).__init__()
        self.efficientnet = EfficientNet.from_name('efficientnet-b0',
                                                   in_channels=shape[1],
                                                   include_top=True,
                                                   batch_norm_momentum=0.1)
        self.lstm = nn.LSTM(1000, 1000)
        self.fc = nn.Linear(in_features=1000, out_features=1)

    def forward(self, x):
        """

        """
        # encoder
        x = torch.unbind(x, dim=1)
        enconder_out = []
        for item in x:
            x_encoder = self.efficientnet(item)
            enconder_out.append(x_encoder.unsqueeze(0))
        enconder_out = torch.cat(enconder_out, dim=0).permute(1, 0, -1)
        # LSTM
        lstm_out, _ = self.lstm(enconder_out)
        # decoder
        output = F.relu(self.fc(lstm_out[:, -1, :]))
        return output
