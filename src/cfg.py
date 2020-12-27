"""

"""
cfg = {
    # paths preprocess
    "PATH_VIDEO": "/home/ditu/Documents/03_Projects/speed_challenge/data/raw_data/train.mp4",
    "PATH_LABELS_TXT": "/home/ditu/Documents/03_Projects/speed_challenge/data/raw_data/train.txt",
    # paths train
    "PATH_FRAMES": "/home/ditu/Documents/03_Projects/speed_challenge/data/frames",
    "PATH_LABELS": "/home/ditu/Documents/03_Projects/speed_challenge/data/train_labels.npy",
    "PATH_SAVE_MODEL": "/home/ditu/Documents/03_Projects/speed_challenge/models/model_SGD_Momentum.pth",
    # paths predict
    "PATH_VIDEO_PREDICT": "/home/ditu/Documents/03_Projects/speed_challenge/data/raw_data/test.mp4",
    "PATH_LABELS_PREDICT": None,
    "PATH_SAVE_RESULTS": "/home/ditu/Documents/03_Projects/speed_challenge/results/results_test_video.pkl",

    # preprocess
    "PREPROCESS_DATASET": False,

    # train model
    "TRAIN_MODEL": False,
    "NUM_SAMPLES_TRAIN": None,  # if None all the data available (whole video) will be taken
    "INPUT_SHAPE": (3, 1, 224, 224),
    "NUM_DATA_SPLITS": 32,
    "TRAIN_VAL_SPLIT": 0.8,
    "BATCH_SIZE": 15,
    "EPOCHS": 30,
    "NUM_WORKERS": 4,
    "LEARNING_RATE": 0.001,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 0.001,

    # make prediction
    "PREDICT_RESULTS": False,
    "NUM_SAMPLES_PREDICTION": None,  # if None all the data available (whole video) will be taken
    "PT1_TIME_CONSTANT": 2,

    # plot results
    "PLOT_RESULTS": True,
    "SHOW_VIDEO": False,
}
