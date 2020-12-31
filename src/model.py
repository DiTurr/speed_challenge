"""

"""
import torch
import torch.nn as nn
import torch.nn.functional as F  # NOQA
from efficientnet_pytorch import EfficientNet
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.preprocess import VideoDataset
from src.postprocess import low_pass_filter


class SpeedChallengeModel:
    def __init__(self, input_shape):
        """

        """
        self.model = EfficientNetConvLSTM(input_shape)
        if torch.cuda.is_available():
            self.model.cuda()
        self.input_shape = input_shape
        self.epochs = None
        self.training_generator = None
        self.validation_generator = None
        self.optimizer = None
        self.loss_function = None
        self.history = None

    def train(self, epochs, training_generator, validation_generator, optimizer, loss_function, path_save_model):
        """

        """
        # set attributes
        self.epochs = epochs
        self.training_generator = training_generator
        self.validation_generator = validation_generator
        self.optimizer = optimizer
        self.loss_function = loss_function

        # CUDA for PyTorch
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        # loop over epochs
        self.history = {"loss": [], "val_loss": []}
        loss_val_best = float("inf")
        for epoch in range(self.epochs):
            # create progress bar
            pbar = tqdm(total=len(self.training_generator) + len(self.validation_generator),
                        ncols=100, dynamic_ncols=True,
                        desc=str(epoch + 1).zfill(5) + "/" + str(self.epochs).zfill(5) + ": ")
            # train epoch
            mean_loss_epoch_train, mean_loss_epoch_val = self.train_epoch(device, pbar)

            # save history information
            if mean_loss_epoch_train is not None:
                self.history["loss"].append(mean_loss_epoch_train)
            if mean_loss_epoch_val is not None:
                self.history["val_loss"].append(mean_loss_epoch_val)

            # close progress bar
            pbar.close()

            # check if val loss has improved and save model if so
            if mean_loss_epoch_val < loss_val_best:
                print("[INFO] Validation loss improved from " + str(loss_val_best) + " to " + str(mean_loss_epoch_val))
                loss_val_best = mean_loss_epoch_val
                self.save(path_save_model=path_save_model.split('.')[0] +
                                          "_epoch_" + str(epoch) + # NOQA
                                          "." + path_save_model.split('.')[1])

        # ensure progress bar is closed
        pbar.close()  # NOQA

        # save model
        print("[INFO] Saving model ... ")
        self.save(path_save_model=path_save_model)

    def train_epoch(self, device, pbar):
        """

        """
        # training
        running_loss_batch_train = 0
        mean_loss_batch_train = None
        with torch.set_grad_enabled(True):
            for index, (x_batch, y_speed_batch) in enumerate(self.training_generator):
                # transfer to GPU
                x_batch = x_batch.to(device)
                y_speed_batch = y_speed_batch.to(device)

                # set model to training mode
                self.model.train()

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                y_hat_speed_batch = self.model(x_batch)
                loss_batch_train = self.loss_function(y_hat_speed_batch, y_speed_batch)

                # calculate total loss
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
            for index, (x_batch, y_speed_batch) in enumerate(self.validation_generator):
                # transfer to GPU
                x_batch = x_batch.to(device)
                y_speed_batch = y_speed_batch.to(device)

                # set model to evaluate model and do forward computation
                self.model.eval()
                y_hat_speed_batch = self.model(x_batch)
                # loss function for the speed
                loss_batch_val = self.loss_function(y_hat_speed_batch, y_speed_batch)

                # printing information/statistics
                running_loss_batch_val += loss_batch_val.item()
                mean_loss_batch_val = running_loss_batch_val / (index + 1)
                pbar.update(1)
                pbar.set_postfix_str("Loss: {:.4f}; Validation Loss: {:.4f}".
                                     format(mean_loss_batch_train, mean_loss_batch_val))

        # return value
        return mean_loss_batch_train, mean_loss_batch_val

    def predict(self, path_video, path_labels, num_samples=None, time_constant_filter=1):
        """

        """
        # preprocess function inputs
        # generate video capture and calculate number of frames to be predicted
        video_capture = cv.VideoCapture(path_video)
        if num_samples is None:
            num_total_frames = int(video_capture.get(cv.CAP_PROP_FRAME_COUNT))
        else:
            num_total_frames = num_samples

        # save labels as numpy array (if specified)
        if path_labels is not None:
            speed = np.loadtxt(path_labels)[0:num_total_frames]
        else:
            print("[WARNING] No label data found ... ")
            speed = np.zeros(num_total_frames)
        speed = np.array(speed).reshape((-1, 1))

        # CUDA for PyTorch
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        # loop through samples
        x = []
        speed_hat = []
        for _ in tqdm(range(num_total_frames)):
            # read in image
            success, frame = video_capture.read()
            # to grayscale
            if self.input_shape[1] == 1:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                frame = frame[:, :, np.newaxis]
            # center croping
            frame = VideoDataset.center_crop(frame, self.input_shape)
            # normalize
            frame = frame / 256
            # append to list
            x.append(frame)

            # if the list if bigger than the actual number of frames, prediction can be done
            if len(x) >= self.input_shape[0]:
                # select last three images of the list
                x = x[-self.input_shape[0]:]
                input_model = np.swapaxes(np.asarray(x, dtype=np.float32), 1, 3)
                input_model = input_model[np.newaxis, :, :, :]

                # convert to numpy array and move to GPU
                input_model = torch.from_numpy(input_model)
                input_model = input_model.to(device)

                # set model to evaluate model and do forward computation
                self.model.eval()
                speed_prediction = self.model(input_model)
                speed_prediction = speed_prediction.cpu().detach().numpy().reshape((-1, 1))

                # append results
                speed_hat.append(speed_prediction)

        # make sure that the number of predictions are equal to the number of frames
        # this may happend because more than one frame is needed to calculate the speed
        for _ in range(num_total_frames - len(speed_hat)):
            speed_hat.insert(0, speed_hat[0])

        # transform to numpy and filter speed
        speed_hat = np.array(speed_hat).reshape((-1, 1))
        speed_hat_filter = low_pass_filter(speed_hat, time_constant=time_constant_filter)

        # return
        assert speed.shape[0] == speed_hat.shape[0] == num_total_frames
        return speed, speed_hat, speed_hat_filter

    def load(self, path_model):
        """

        """
        checkpoint = torch.load(path_model)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.epochs = checkpoint["epoch"]
        self.history = checkpoint["history"]

    def save(self, path_save_model):
        """

        """
        torch.save({
            "epoch": self.epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }, path_save_model)

    def plot_history(self):
        """

        """
        if self.history is not None:
            fig, axs = plt.subplots()
            axs.plot(self.history["loss"], label="loss")
            axs.plot(self.history["val_loss"], label="val_loss")
            axs.set_title("Losses")
            axs.set_xlim(0, None)
            axs.set_ylim(0, 16)
            plt.legend()
            plt.grid()
        else:
            print("[ERROR] No history to plot ...")

    @staticmethod
    def plot_prediction(y, y_hat, labels):
        """

        """
        fig, axs = plt.subplots()
        for idx, y_hat_actual in enumerate(y_hat):
            axs.plot(y_hat_actual, label=labels[idx])
        if y is not None:
            axs.plot(y, label="y")
        axs.set_title("Speed Prediction")
        axs.set_xlim(0, None)
        axs.set_ylim(0, 30)
        plt.legend()
        plt.grid()


class EfficientNetConvLSTM(nn.Module):
    def __init__(self, shape, num_classes=1024):
        """

        """
        super(EfficientNetConvLSTM, self).__init__()
        self.efficientnet = EfficientNet.from_name('efficientnet-b0',
                                                   in_channels=shape[1],
                                                   include_top=True,
                                                   batch_norm_momentum=0.1,
                                                   num_classes=num_classes)
        self.lstm = nn.LSTM(num_classes, num_classes)
        self.fc_speed = nn.Linear(in_features=num_classes, out_features=1)

    def forward(self, x):
        """

        """
        # encoder
        x = torch.unbind(x, dim=1)
        enconder_out = []
        for item in x:
            x_encoder = self.efficientnet(item)
            x_encoder = F.dropout(x_encoder, p=0.5)
            enconder_out.append(x_encoder.unsqueeze(0))
        enconder_out = torch.cat(enconder_out, dim=0).permute(1, 0, -1)
        # LSTM
        lstm_out, _ = self.lstm(enconder_out)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = F.dropout(lstm_out, p=0.5)
        # decoder
        output_speed = F.relu(self.fc_speed(lstm_out))
        return output_speed
