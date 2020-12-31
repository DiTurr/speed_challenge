"""

"""
cfg = {
    # paths preprocess
    "PATH_VIDEO_TRAIN": "/home/ditu/Documents/03_Projects/speed_challenge/data/raw_data/train.mp4",
    "PATH_LABELS_TRAIN": "/home/ditu/Documents/03_Projects/speed_challenge/data/raw_data/train.txt",
    "PATH_FRAMES_TRAIN": "/home/ditu/Documents/03_Projects/speed_challenge/data/frames",

    # paths train model
    "PATH_SAVE_MODEL": "/home/ditu/Documents/03_Projects/speed_challenge/models/model_epoch_42.pth",

    # paths predict and plot results
    "PATH_VIDEO_VAL": "/home/ditu/Documents/03_Projects/speed_challenge/data/raw_data/test.mp4",
    "PATH_LABELS_VAL": None,
    "PATH_SAVE_VAL": "/home/ditu/Documents/03_Projects/speed_challenge/results/results_test_video.txt",

    # preprocess
    "PREPROCESS_DATASET": False,

    # train model
    "TRAIN_MODEL": False,
    "NUM_SAMPLES_TRAIN": None,  # if None all the data available (whole video) will be taken
    "INPUT_SHAPE": (3, 1, 224, 224),
    "NUM_DATA_SPLITS": 32,
    "TRAIN_VAL_SPLIT": 0.8,
    "BATCH_SIZE": 15,
    "EPOCHS": 50,
    "NUM_WORKERS": 4,
    "LEARNING_RATE": 0.001,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 0.001,

    # make prediction and plot results
    "PREDICT_RESULTS": False,
    "PT1_TIME_CONSTANT": 2,
    "PLOT_RESULTS": True,
    "SHOW_VIDEO": True,
}
