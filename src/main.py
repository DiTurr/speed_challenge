"""

"""
import torch
from torch.utils.data import DataLoader  # NOQA
# from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import pickle

from src.preprocess import VideoDataset
from src.model import SpeedChallengeModel

cfg = {
    # preprocess
    "PREPROCESS_DATASET": False,

    # train model
    "TRAIN_MODEL": True,
    "PATH_FRAMES": "/home/ditu/Documents/03_Projects/speed_challenge/data/frames",
    "PATH_LABELS": "/home/ditu/Documents/03_Projects/speed_challenge/data/labels.npy",
    "INPUT_SHAPE": (3, 3, 224, 224),
    "NUM_SAMPLES": None,  # if None all the data available (whole video) will be taken
    "DATA_SPLITS": 16,
    "TRAIN_VAL_SPLIT": 0.8,
    "BATCH_SIZE": 15,
    "EPOCHS": 15,
    "NUM_WORKERS": 1,
    "LEARNING_RATE": 0.001,
    "NORMALIZE_OUTPUT": False,
    "PATH_SAVE_MODEL": "/home/ditu/Documents/03_Projects/speed_challenge/models/model.pth",

    # make prediction
    "PREDICT_RESULTS": True,
    "PATH_PREDICT_VIDEO": "/home/ditu/Documents/03_Projects/speed_challenge/data/raw_data/train.mp4",
    "NUM_SAMPLES_PREDICTION": 3000,
    "PT1_TIME_CONSTANT": 1,
    "PATH_SAVE_RESULTS": "/home/ditu/Documents/03_Projects/speed_challenge/results_train_video.pkl",

    # plot results
    "PLOT_RESULTS": True,
}


def main():
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
            path_frames=cfg["PATH_FRAMES"],
            data_splits=cfg["DATA_SPLITS"],
            num_samples=cfg["NUM_SAMPLES"],
            num_input_frames=cfg["INPUT_SHAPE"][0],
            train_val_split=cfg["TRAIN_VAL_SPLIT"],
            shuffle=False)

        # create training dataloader
        print("[INFO] Creating training dataloader ... ")
        num_train_samples = None if cfg["NUM_SAMPLES"] is None else len(idx_img_train_set)
        dataset_train = VideoDataset(path_frames=cfg["PATH_FRAMES"],
                                     idx_frames=idx_img_train_set,
                                     path_labels=cfg["PATH_LABELS"],
                                     input_shape=cfg["INPUT_SHAPE"],
                                     num_samples=num_train_samples,
                                     normalize_ouput=cfg["NORMALIZE_OUTPUT"],
                                     data_augmentation=True,
                                     stride=1)
        dataloader_train = DataLoader(dataset_train,
                                      batch_size=cfg["BATCH_SIZE"],
                                      shuffle=True,
                                      num_workers=cfg["NUM_WORKERS"],
                                      drop_last=False)

        # create validation dataloader
        print("[INFO] Creating validation dataloader ... ")
        num_test_samples = None if cfg["NUM_SAMPLES"] is None else len(idx_img_test_set)
        dataset_test = VideoDataset(path_frames=cfg["PATH_FRAMES"],
                                    idx_frames=idx_img_test_set,
                                    path_labels=cfg["PATH_LABELS"],
                                    input_shape=cfg["INPUT_SHAPE"],
                                    num_samples=num_test_samples,
                                    normalize_ouput=cfg["NORMALIZE_OUTPUT"],
                                    data_augmentation=False,
                                    stride=1)
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
                    optimizer=torch.optim.Adam(model.model.parameters(), lr=cfg["LEARNING_RATE"]),
                    loss_function=torch.nn.MSELoss(reduction='mean'))

        # save model
        print("[INFO] Saving model ... ")
        model.save(path_save_model=cfg["PATH_SAVE_MODEL"])

    if cfg["PREDICT_RESULTS"]:
        # predict
        print("[INFO] Predicting speed ... ")
        model = SpeedChallengeModel(input_shape=cfg["INPUT_SHAPE"])
        model.load(path_model=cfg["PATH_SAVE_MODEL"])
        y, y_hat, y_hat_filter = model.predict(
            path_video=cfg["PATH_PREDICT_VIDEO"],
            path_labels=cfg["PATH_LABELS"],
            num_samples=cfg["NUM_SAMPLES_PREDICTION"],
            normalize_ouput=cfg["NORMALIZE_OUTPUT"],
            time_constant_filter=cfg["PT1_TIME_CONSTANT"])
        print("[INFO] Mean Squared Error for y_hat       : " + str(np.square(np.subtract(y, y_hat)).mean()))
        print("[INFO] Mean Squared Error for y_hat_filter: " + str(np.square(np.subtract(y, y_hat_filter)).mean()))

        # save results
        with open(cfg["PATH_SAVE_RESULTS"], 'wb') as fp:
            pickle.dump({"y": y, "y_hat": y_hat, "y_hat_filter": y_hat_filter}, fp)

    if cfg["PLOT_RESULTS"]:
        # plot training history
        print("[INFO] Plotting history training and results ... ")
        model = SpeedChallengeModel(input_shape=cfg["INPUT_SHAPE"])
        model.load(path_model=cfg["PATH_SAVE_MODEL"])
        model.plot_history()

        # plot prediction
        print("[INFO] Plotting history training and results ... ")
        with open(cfg["PATH_SAVE_RESULTS"], 'rb') as fp:
            results = pickle.load(fp)
        model.plot_prediction(results["y"], [results["y_hat"], results["y_hat_filter"]],
                              labels=["y_hat", "y_hat_filter"])

        # show figures
        plt.show()


if __name__ == "__main__":
    main()
