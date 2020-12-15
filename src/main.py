"""

"""
import torch
from torch.utils.data import DataLoader  # NOQA
from torchsummary import summary
import numpy as np

from src.preprocess import VideoDataset, low_pass_filter
from src.model import SpeedChallengeModel

cfg = {
    "PREPROCESS_VIDEO": False,
    "NUM_SAMPLES": None,
    "INPUT_SHAPE": (3, 3, 224, 224),
    "BATCH_SIZE": 15,
    "NUM_EPOCHS": 30,
    "NUM_WORKERS": 4,
    "LEARNING_RATE": 0.001,
    "NORMALIZE_OUTPUT": False,
    "PT1_TIME_CONSTANT": 2,
}

if __name__ == "__main__":
    # preprocess video
    if cfg["PREPROCESS_VIDEO"]:
        print("[INFO] Preprocessing dataset ... ")
        VideoDataset.save_video_to_frames(path_video="/home/ditu/Documents/03_Projects/speed_challenge/data/train.mp4",
                                          path_label="/home/ditu/Documents/03_Projects/speed_challenge/data/train.txt",
                                          path_save_img="/home/ditu/Documents/03_Projects/speed_challenge/data/",
                                          train_val_split=0.8)

    # create training  dataloader
    print("[INFO] Creating training dataloader ... ")
    dataset_train = VideoDataset("/home/ditu/Documents/03_Projects/speed_challenge/data/train_set",
                                 "/home/ditu/Documents/03_Projects/speed_challenge/data/train_label.npy",
                                 input_shape=cfg["INPUT_SHAPE"],
                                 num_samples=cfg["NUM_SAMPLES"],
                                 normalize_ouput=cfg["NORMALIZE_OUTPUT"])
    dataloader_train = DataLoader(dataset_train,
                                  batch_size=cfg["BATCH_SIZE"],
                                  shuffle=True,
                                  num_workers=cfg["NUM_WORKERS"],
                                  drop_last=False)

    # create validation dataloader
    print("[INFO] Creating validation dataloader ... ")
    dataset_test = VideoDataset("/home/ditu/Documents/03_Projects/speed_challenge/data/test_set",
                                "/home/ditu/Documents/03_Projects/speed_challenge/data/test_label.npy",
                                input_shape=cfg["INPUT_SHAPE"],
                                num_samples=cfg["NUM_SAMPLES"],
                                normalize_ouput=cfg["NORMALIZE_OUTPUT"])
    dataloader_test = DataLoader(dataset_test,
                                 batch_size=cfg["BATCH_SIZE"],
                                 shuffle=False,
                                 num_workers=cfg["NUM_WORKERS"],
                                 drop_last=False)

    # create model
    print("[INFO] Creating model ... ")
    model = SpeedChallengeModel(input_shape=cfg["INPUT_SHAPE"])
    # summary(model.model, cfg["INPUT_SHAPE"])

    # train model
    print("[INFO] Training model ... ")
    model.train(num_epochs=cfg["NUM_EPOCHS"],
                training_generator=dataloader_train,
                validation_generator=dataloader_test,
                optimizer=torch.optim.SGD(model.model.parameters(), lr=cfg["LEARNING_RATE"]),
                loss_function=torch.nn.MSELoss(reduction='mean'))

    # predict
    print("[INFO] Predicting speed ... ")
    y, y_hat = model.predict(
        path_save_img="/home/ditu/Documents/03_Projects/speed_challenge/data/train_set",
        path_label_speed="/home/ditu/Documents/03_Projects/speed_challenge/data/train_label.npy",
        batch_size=cfg["BATCH_SIZE"],
        num_samples=cfg["NUM_SAMPLES"],
        normalize_ouput=cfg["NORMALIZE_OUTPUT"])
    y_hat_filter = low_pass_filter(y_hat, time_constant=cfg["PT1_TIME_CONSTANT"])
    print("[INFO] Mean Squared Error for y_hat       : " + str(np.square(np.subtract(y, y_hat)).mean()))
    print("[INFO] Mean Squared Error for y_hat_filter: " + str(np.square(np.subtract(y, y_hat_filter)).mean()))

    # predict
    print("[INFO] Plotting history training and results ... ")
    model.plot_history()
    model.plot_prediction(y, [y_hat, y_hat_filter], labels=["y_hat", "y_hat_filter"])
    model.plot_show()
