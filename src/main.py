"""

"""
import torch
from torch.utils.data import DataLoader  # NOQA
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import pickle

from src.preprocess import VideoDataset
from src.model import SpeedChallengeModel
from src.postprocess import show_video
from src.cfg import cfg


def main():
    if cfg["PREPROCESS_DATASET"]:
        # preprocess video
        print("[INFO] Preprocessing dataset ... ")
        VideoDataset.save_video_to_frames(
            path_video=cfg["PATH_VIDEO"],
            path_labels=cfg["PATH_LABELS_TXT"],
            path_save_frames=cfg["PATH_FRAMES"],
            path_save_labels=cfg["PATH_LABELS"])

    if cfg["TRAIN_MODEL"]:
        # create dataset split
        print("[INFO] Creating dataset split ... ")
        idx_img_train_set, idx_img_test_set = VideoDataset.split_train_val_set(
            path_frames=cfg["PATH_FRAMES"],
            num_data_splits=cfg["NUM_DATA_SPLITS"],
            num_samples=cfg["NUM_SAMPLES_TRAIN"],
            num_input_frames=cfg["INPUT_SHAPE"][0],
            train_val_split=cfg["TRAIN_VAL_SPLIT"],
            shuffle=False)

        # create training dataloader
        print("[INFO] Creating training dataloader ... ")
        num_train_samples = None if cfg["NUM_SAMPLES_TRAIN"] is None else len(idx_img_train_set)
        dataset_train = VideoDataset(path_frames=cfg["PATH_FRAMES"],
                                     idx_frames=idx_img_train_set,
                                     path_labels=cfg["PATH_LABELS"],
                                     input_shape=cfg["INPUT_SHAPE"],
                                     num_samples=num_train_samples,
                                     data_augmentation=True)
        dataloader_train = DataLoader(dataset_train,
                                      batch_size=cfg["BATCH_SIZE"],
                                      num_workers=cfg["NUM_WORKERS"],
                                      shuffle=True,
                                      drop_last=False)

        # create validation dataloader
        print("[INFO] Creating validation dataloader ... ")
        num_test_samples = None if cfg["NUM_SAMPLES_TRAIN"] is None else len(idx_img_test_set)
        dataset_test = VideoDataset(path_frames=cfg["PATH_FRAMES"],
                                    idx_frames=idx_img_test_set,
                                    path_labels=cfg["PATH_LABELS"],
                                    input_shape=cfg["INPUT_SHAPE"],
                                    num_samples=num_test_samples,
                                    data_augmentation=False)
        dataloader_test = DataLoader(dataset_test,
                                     batch_size=cfg["BATCH_SIZE"],
                                     num_workers=cfg["NUM_WORKERS"],
                                     shuffle=True,
                                     drop_last=False)

        # create model
        print("[INFO] Creating model ... ")
        model = SpeedChallengeModel(input_shape=cfg["INPUT_SHAPE"])
        summary(model.model, cfg["INPUT_SHAPE"])

        # train model
        print("[INFO] Training model ... ")
        model.train(epochs=cfg["EPOCHS"],
                    training_generator=dataloader_train,
                    validation_generator=dataloader_test,
                    optimizer=torch.optim.SGD(model.model.parameters(),
                                              lr=cfg["LEARNING_RATE"],
                                              momentum=cfg["MOMENTUM"],
                                              nesterov=False,
                                              weight_decay=cfg["WEIGHT_DECAY"]),
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
            path_video=cfg["PATH_VIDEO_PREDICT"],
            path_labels=cfg["PATH_LABELS_PREDICT"],
            num_samples=None,
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
        print("[INFO] Plotting predictions ... ")
        with open(cfg["PATH_SAVE_RESULTS"], 'rb') as fp:
            results = pickle.load(fp)
        model.plot_prediction(results["y"], [results["y_hat"], results["y_hat_filter"]],
                              labels=["y_hat", "y_hat_filter"])
        model.plot_prediction(results["y"], [results["y_hat_filter"]], labels=["y_hat_filter"])

        # plot dataset histogram
        print("[INFO] Plotting histogram ... ")
        idx_img_train_set, idx_img_test_set = VideoDataset.split_train_val_set(
            path_frames=cfg["PATH_FRAMES"],
            num_data_splits=cfg["NUM_DATA_SPLITS"],
            num_samples=cfg["NUM_SAMPLES_TRAIN"],
            num_input_frames=cfg["INPUT_SHAPE"][0],
            train_val_split=cfg["TRAIN_VAL_SPLIT"],
            shuffle=False)
        VideoDataset.plot_labels_histogram(cfg["PATH_LABELS"], idx_img_train_set, idx_img_test_set)

        # show video
        if cfg["SHOW_VIDEO"]:
            show_video(cfg["PATH_VIDEO_PREDICT"],
                       results["y"],
                       results["y_hat_filter"],
                       input_frames=cfg["INPUT_SHAPE"][0])

        # show figures
        plt.show()


if __name__ == "__main__":
    main()
