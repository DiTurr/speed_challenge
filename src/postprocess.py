"""

"""
import numpy as np
import cv2 as cv


def low_pass_filter(x, y_init=None, fps=30, time_constant=1):
    """

    """
    # parameters of the low pass filter
    lowpass_cutoff = 1 / time_constant
    dt = 1 / fps
    rc = 1 / (2 * np.pi * lowpass_cutoff)
    alpha = dt / (rc + dt)

    # create output array
    y = np.zeros_like(x)

    # loop through all numbers to filter
    if y_init is None:
        y[0] = x[0]
    else:
        y[0] = y_init
    for idx in range(1, x.shape[0]):
        y[idx] = x[idx] * alpha + (1 - alpha) * y[idx - 1]

    # return output
    return y


def show_video(path_video, y, y_hat, input_frames=1):
    """

    """
    video_capture = cv.VideoCapture(path_video)
    num_total_frames = min(int(video_capture.get(cv.CAP_PROP_FRAME_COUNT)), y.shape[0], y_hat.shape[0])
    for idx_frame in range(num_total_frames):
        success, frame = video_capture.read()

        # display the resulting frame
        if success:
            if idx_frame >= (input_frames-1):
                # inserting text on video
                cv.putText(frame, "Y: " + str(y[idx_frame-input_frames + 1]),
                           (50, 50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv.putText(frame, "Y_HAT: " + str(y_hat[idx_frame-input_frames + 1]),
                           (50, 100), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                # plot rectangle input to model
                frame = cv.rectangle(frame, (208, 128), (208+224, 128+224), (255, 0, 0), 1)
                # img = img[196:-196, 232:-232]
                frame = cv.rectangle(frame, (232, 196), (232 + 176, 196 + 88), (255, 255, 0), 1)
            cv.imshow('Frame', frame)
            # press Q on keyboard to  exit
            if cv.waitKey(50) & 0xFF == ord('q'):
                break

        # break the loop
        else:
            break
