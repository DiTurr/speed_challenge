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

# Proposed model:
To solve the challenge the following solution has been proposed:
 - 3 frames are used for the prediction.
 - Encoder based on EfficientNet B0.
 - Encoder is fed into LSTM layer.
 - LSTM output is fed into a fully connected layer.
 - Model output is filter with a first order loss pass filter (PT1). For implementation, see postprocessing.py
 - For the training, just the video test.mp4 has been used.

# Results:
### Training history:

<img src="https://github.com/DiTurr/speed_challenge/blob/main/results/history_training.png" height="300" width="400" />

### Results on training dataset:

<img src="https://github.com/DiTurr/speed_challenge/blob/main/results/results_train.png" height="300" width="400" />

### Results on validation dataset:

...


# Conclusions:
...
