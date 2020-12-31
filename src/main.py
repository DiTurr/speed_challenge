"""

"""
import torch
from torch.utils.data import DataLoader  # NOQA
# from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt

from src.preprocess import VideoDataset
from src.model import SpeedChallengeModel
from src.postprocess import show_video
from src.cfg import cfg


def preprocess_dataset():
    """

    """
    # preprocess video
    print("[INFO] Preprocessing dataset ... ")
    VideoDataset.save_video_to_frames(
        path_video=cfg["PATH_VIDEO_TRAIN"],
        path_save_frames=cfg["PATH_FRAMES_TRAIN"])


def train_model():
    """

    """
    # create dataset split
    print("[INFO] Creating dataset split ... ")
    idx_img_train_set, idx_img_test_set = VideoDataset.split_train_val_set(
        path_frames=cfg["PATH_FRAMES_TRAIN"],
        num_data_splits=cfg["NUM_DATA_SPLITS"],
        num_samples=cfg["NUM_SAMPLES_TRAIN"],
        num_input_frames=cfg["INPUT_SHAPE"][0],
        train_val_split=cfg["TRAIN_VAL_SPLIT"],
        shuffle=False)

    # create training dataloader
    print("[INFO] Creating training dataloader ... ")
    num_train_samples = None if cfg["NUM_SAMPLES_TRAIN"] is None else len(idx_img_train_set)
    dataset_train = VideoDataset(path_frames=cfg["PATH_FRAMES_TRAIN"],
                                 idx_frames=idx_img_train_set,
                                 path_labels=cfg["PATH_LABELS_TRAIN"],
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
    dataset_test = VideoDataset(path_frames=cfg["PATH_FRAMES_TRAIN"],
                                idx_frames=idx_img_test_set,
                                path_labels=cfg["PATH_LABELS_TRAIN"],
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
    # summary(model.model, cfg["INPUT_SHAPE"])

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
                loss_function=torch.nn.MSELoss(reduction='mean'),
                path_save_model=cfg["PATH_SAVE_MODEL"])


def predict_results():
    """

    """
    # predict
    print("[INFO] Predicting speed ... ")
    model = SpeedChallengeModel(input_shape=cfg["INPUT_SHAPE"])
    model.load(path_model=cfg["PATH_SAVE_MODEL"])
    y, y_hat, y_hat_filter = model.predict(
        path_video=cfg["PATH_VIDEO_VAL"],
        path_labels=cfg["PATH_LABELS_VAL"],
        num_samples=None,
        time_constant_filter=cfg["PT1_TIME_CONSTANT"])
    print("[INFO] Mean Squared Error for y_hat       : " + str(np.square(np.subtract(y, y_hat)).mean()))
    print("[INFO] Mean Squared Error for y_hat_filter: " + str(np.square(np.subtract(y, y_hat_filter)).mean()))

    # save results
    np.savetxt(cfg["PATH_SAVE_VAL"], y_hat_filter)


def plot_results():
    """

    """
    # plot training history
    print("[INFO] Plotting history training and results ... ")
    model = SpeedChallengeModel(input_shape=cfg["INPUT_SHAPE"])
    model.load(path_model=cfg["PATH_SAVE_MODEL"])
    model.plot_history()

    # plot prediction
    print("[INFO] Plotting predictions ... ")
    y = np.loadtxt(cfg["PATH_LABELS_VAL"]) if cfg["PATH_LABELS_VAL"] is not None else None
    y_hat = np.loadtxt(cfg["PATH_SAVE_VAL"])
    model.plot_prediction(y, [y_hat], labels=["y_hat"])

    # show video
    if cfg["SHOW_VIDEO"]:
        show_video(cfg["PATH_VIDEO_VAL"], y, y_hat)

    # show figures
    plt.show()


def main():
    if cfg["PREPROCESS_DATASET"]:
        preprocess_dataset()

    if cfg["TRAIN_MODEL"]:
        train_model()

    if cfg["PREDICT_RESULTS"]:
        predict_results()

    if cfg["PLOT_RESULTS"]:
        plot_results()


if __name__ == "__main__":
    main()
