"""

"""
import torch
from torch.utils.data import DataLoader  # NOQA
# from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt

from src.preprocess import VideoDataset
from src.model import SpeedChallengeModel

cfg = {
    "PREPROCESS_DATASET": False,
    "TRAIN_MODEL": False,
    "PREDICT_SPEED": True,
    "INPUT_SHAPE": (3, 3, 224, 224),
    "NUM_SAMPLES_TRAIN": None,
    "NUM_SAMPLES_TEST": None,
    "NUM_SAMPLES_PREDICTION": None,
    "BATCH_SIZE": 1,
    "EPOCHS": 15,
    "NUM_WORKERS": 4,
    "LEARNING_RATE": 0.001,
    "NORMALIZE_OUTPUT": False,
    "PT1_TIME_CONSTANT": 1,
}

if __name__ == "__main__":
    if cfg["PREPROCESS_DATASET"]:
        # preprocess video
        print("[INFO] Preprocessing dataset ... ")
        VideoDataset.save_video_to_frames(
            path_video="/home/ditu/Documents/03_Projects/speed_challenge/data/train.mp4",
            path_labels="/home/ditu/Documents/03_Projects/speed_challenge/data/train.txt",
            path_save_frames="/home/ditu/Documents/03_Projects/speed_challenge/data/",
            path_save_labels="/home/ditu/Documents/03_Projects/speed_challenge/data/labels.npy")

    if cfg["TRAIN_MODEL"]:
        # create dataset split
        print("[INFO] Creating dataset split ... ")
        idx_img_train_set, idx_img_test_set = VideoDataset.split_train_test_set(
            path_frames="/home/ditu/Documents/03_Projects/speed_challenge/data/frames",
            num_frames=cfg["INPUT_SHAPE"][0],
            train_val_split=0.8)

        # create training  dataloader
        print("[INFO] Creating training dataloader ... ")
        dataset_train = VideoDataset(path_frames="/home/ditu/Documents/03_Projects/speed_challenge/data/frames",
                                     idx_frames=idx_img_train_set,
                                     path_labels="/home/ditu/Documents/03_Projects/speed_challenge/data/labels.npy",
                                     input_shape=cfg["INPUT_SHAPE"],
                                     num_samples=cfg["NUM_SAMPLES_TRAIN"],
                                     normalize_ouput=cfg["NORMALIZE_OUTPUT"])
        dataloader_train = DataLoader(dataset_train,
                                      batch_size=cfg["BATCH_SIZE"],
                                      shuffle=True,
                                      num_workers=cfg["NUM_WORKERS"],
                                      drop_last=False)

        # create validation dataloader
        print("[INFO] Creating validation dataloader ... ")
        dataset_test = VideoDataset(path_frames="/home/ditu/Documents/03_Projects/speed_challenge/data/frames",
                                    idx_frames=idx_img_test_set,
                                    path_labels="/home/ditu/Documents/03_Projects/speed_challenge/data/labels.npy",
                                    input_shape=cfg["INPUT_SHAPE"],
                                    num_samples=cfg["NUM_SAMPLES_TEST"],
                                    normalize_ouput=cfg["NORMALIZE_OUTPUT"])
        dataloader_test = DataLoader(dataset_test,
                                     batch_size=cfg["BATCH_SIZE"],
                                     shuffle=True,
                                     num_workers=cfg["NUM_WORKERS"],
                                     drop_last=False)

        # create model
        print("[INFO] Creating model ... ")
        model = SpeedChallengeModel(input_shape=cfg["INPUT_SHAPE"])
        # summary(model.model, cfg["INPUT_SHAPE"])

        # train model
        print("[INFO] Training model ... ")
        model.train(epochs=cfg["EPOCHS"],
                    training_generator=dataloader_train,
                    validation_generator=dataloader_test,
                    optimizer=torch.optim.SGD(model.model.parameters(), lr=cfg["LEARNING_RATE"]),
                    loss_function=torch.nn.MSELoss(reduction='mean'))

        # save model
        print("[INFO] Saving model ... ")
        model.save(path_save_model="/home/ditu/Documents/03_Projects/speed_challenge/models/model.pth")

        # plot training history
        print("[INFO] Plotting history training and results ... ")
        model.plot_history()

    if cfg["PREDICT_SPEED"]:
        # predict
        print("[INFO] Predicting speed ... ")
        model = SpeedChallengeModel(input_shape=cfg["INPUT_SHAPE"])
        model.load(path_model="/home/ditu/Documents/03_Projects/speed_challenge/models/model.pth")
        y, y_hat, y_hat_filter = model.predict(
            path_video="/home/ditu/Documents/03_Projects/speed_challenge/data/raw_data/train.mp4",
            path_labels="/home/ditu/Documents/03_Projects/speed_challenge/data/labels.npy",
            num_samples=cfg["NUM_SAMPLES_PREDICTION"],
            normalize_ouput=cfg["NORMALIZE_OUTPUT"],
            time_constant_filter=cfg["PT1_TIME_CONSTANT"])
        print("[INFO] Mean Squared Error for y_hat       : " + str(np.square(np.subtract(y, y_hat)).mean()))
        print("[INFO] Mean Squared Error for y_hat_filter: " + str(np.square(np.subtract(y, y_hat_filter)).mean()))

        # plot prediction
        print("[INFO] Plotting history training and results ... ")
        model.plot_prediction(y, [y_hat, y_hat_filter], labels=["y_hat", "y_hat_filter"])

    # show figures
    plt.show()
