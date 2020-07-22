
import cv2
import os
from argparse import ArgumentParser
import numpy as np
import logging
import time
from input_feeder import InputFeeder
from mouse_controller import MouseController
from face_detection import Face_Detection
from facial_landmarks import Facial_Landmarks
from head_pose_estimation import Head_Pose_Estimation
from gaze_estimation import Gaze_Estimation


def get_args():
    parser = ArgumentParser(
        description='Mouse Pointer Controller using eye gaze')
    parser.add_argument("-t", "--input-type", required=True, type=str,
                        help="Type of input (video or cam)")
    parser.add_argument("-i", "--input", required=False, type=str,
                        help="Input file")
    parser.add_argument("-o", "--out", type=str, default=None,
                        help="Output file with the processed content")
    parser.add_argument("-p", "--preview", action='store_true', default=False,
                        help="Should preview face and eyes")
    parser.add_argument("-m", "--model", type=str, default="FP32",
                        help="Model precision to use. One of FP32, FP16 or FP16-INT8")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Device used to process model. One or CPU or GPU")
    return parser.parse_args()


def main(args):
    fd = Face_Detection(
        "models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001")
    start = time.time()
    fd.load_model()
    fd_loaded = time.time() - start

    fl = Facial_Landmarks(
        "models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009")
    fl.load_model()
    hp = Head_Pose_Estimation(
        "models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001")
    hp.load_model()
    gs = Gaze_Estimation(
        "models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002")
    gs.load_model()

    input_feed = InputFeeder(args.input_type, args.input)
    input_feed.load_data()

    mc = MouseController("medium", "fast")

    for frame in input_feed.next_batch():
        if frame is not None:
            # face detection
            face_frame = fd.predict(frame.copy())
            # eye detection through facial landmarks
            left_eye_image, left_x, left_y, right_eye_image, right_x, right_y = fl.predict(
                face_frame)
            # head pose
            yaw, pitch, roll = hp.predict(face_frame)

            gaze_vector = gs.predict(
                left_eye_image, right_eye_image, (yaw, pitch, roll))
            mc.move(gaze_vector[0], gaze_vector[1])

            face_frame = cv2.circle(
                face_frame, (right_x, right_y), 5, (255, 0, 0), -5)
            face_frame = cv2.circle(
                face_frame, (left_x, left_y), 5, (255, 0, 0), -5)
            cv2.putText(face_frame, "yaw:{:.2f} - pitch:{:.2f} - roll:{:.2f}".format(
                yaw, pitch, roll), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1)
            cv2.imshow('left eye', left_eye_image)
            cv2.imshow('right eye', right_eye_image)
            x, y, z = gaze_vector
            cv2.arrowedLine(face_frame, (left_x, left_y), (left_x +
                                                           int(x*100), left_y + int(-y*100)), (0, 0, 255), 2)
            cv2.arrowedLine(face_frame, (right_x, right_y), (right_x +
                                                             int(x*100), right_y + int(-y*100)), (0, 0, 255), 2)
            cv2.waitKey(60)
            cv2.imshow('face detection', face_frame)
        else:
            break

    input_feed.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = get_args()
    main(args)
