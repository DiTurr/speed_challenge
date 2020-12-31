# Introduction: comma.ai speed challenge
Description by [https://github.com/commaai/speedchallenge](https://github.com/commaai/speedchallenge)  

```
Welcome to the comma.ai 2017 Programming Challenge!

Basically, your goal is to predict the speed of a car from a video.

data/train.mp4 is a video of driving containing 20400 frames. Video is shot at 20 fps.
data/train.txt contains the speed of the car at each frame, one speed on each line.

data/test.mp4 is a different driving video containing 10798 frames. Video is shot at 20 fps.
Your deliverable is test.txt

We will evaluate your test.txt using mean squared error. <10 is good. <5 is better. <3 is heart.
```

# Proposed solution:
To solve the challenge the following solution has been proposed:

== Preprocessing ==
 - The train.mp4 video is save into 20400 frames. For implementation, see preprocessing.py (save_video_to_frames) 
 - The train.mp4 video is split into "NUM_DATA_SPLITS" (32) division, which are split again into train and validation dataset "TRAIN_VAL_SPLIT" (0.8). For implementaion, see preprocessing.py (split_train_val_set)

== Inputs ==
 - 3 frames are used for the prediction. At least 2 frames are need due to the fact that v = dx/dt (two frames needed to calculate difference).
 - Grayscale center cropped images are used (224 x 224 x 1) (H x W x C) normalized between 0 and 1.
 
== Arquitecture ==
 - Encoder based on EfficientNet B0.
 - Encoder is fed into LSTM layer.
 - LSTM output is fed into a fully connected layer.
 - Model output is filter with a first order loss pass filter (PT1). For implementation, see postprocessing.py
 
== Training ==
 - For the training, just the video test.mp4 has been used.
 - Data augmentation (brightness, gamma, horizintal flip).
 - L2 regularization (try to reduce overfitting).

# Results:
### Training history:

<img src="https://github.com/DiTurr/speed_challenge/blob/main/results/history_training.png" height="300" width="400" />

### Results on training dataset:

<img src="https://github.com/DiTurr/speed_challenge/blob/main/results/results_train.png" height="300" width="400" />

### Results on validation dataset:

<img src="https://github.com/DiTurr/speed_challenge/blob/main/results/results_test.png" height="300" width="400" />


# Conclusions:
...
