
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
    parser.add_argument("-v", "--visualize", action='store_true', default=False,
                        help="visualize a preview video")
    parser.add_argument("-p", "--precision", type=str, default="FP32",
                        help="Model precision: FP32, FP16 or FP16-INT8")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Device to use CPU or GPU")
    parser.add_argument("-x", "--extensions", type=str, default=None,
                        help="CPU Extensions")
    return parser.parse_args()


def main(args):
    fd = Face_Detection(
        "models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001", args.device, args.extensions)
    start = time.time()
    fd.load_model()
    logging.info(f"------Loading Times {args.precision}------")
    logging.info("Face Detection: {:.5f} sec".format(time.time() - start))

    fl = Facial_Landmarks(
        f"models/intel/landmarks-regression-retail-0009/{args.precision}/landmarks-regression-retail-0009", args.device, args.extensions)
    start = time.time()
    fl.load_model()
    logging.info("Facial Landmarks: {:.5f} sec".format(time.time() - start))
    hp = Head_Pose_Estimation(
        f"models/intel/head-pose-estimation-adas-0001/{args.precision}/head-pose-estimation-adas-0001", args.device, args.extensions)
    start = time.time()
    hp.load_model()
    logging.info("Head Pose Estimation: {:.5f} sec".format(
        time.time() - start))
    gs = Gaze_Estimation(
        f"models/intel/gaze-estimation-adas-0002/{args.precision}/gaze-estimation-adas-0002", args.device, args.extensions)
    start = time.time()
    gs.load_model()
    logging.info("Gaze Estimation: {:.5f} sec".format(time.time() - start))

    input_feed = InputFeeder(args.input_type, args.input)
    input_feed.load_data()

    mc = MouseController("high", "fast")

    inf_time = [0, 0, 0, 0, 0]  # fd, fl, hp, gs, frames
    for frame in input_feed.next_batch():

        if frame is not None:
            inf_time[4] += 1
            # face detection
            start = time.time()
            face_frame = fd.predict(frame.copy())
            inf_time[0] += time.time() - start
            # eye detection through facial landmarks
            start = time.time()
            left_eye_image, left_x, left_y, right_eye_image, right_x, right_y = fl.predict(
                face_frame)
            inf_time[1] += time.time() - start
            # head pose
            start = time.time()
            yaw, pitch, roll = hp.predict(face_frame)
            inf_time[2] += time.time() - start
            # gaze estimation
            start = time.time()
            gaze_vector = gs.predict(
                left_eye_image, right_eye_image, (yaw, pitch, roll))
            inf_time[3] += time.time() - start

            # mouse move
            mc.move(gaze_vector[0], gaze_vector[1])

            if args.visualize:
                face_frame = cv2.circle(
                    face_frame, (right_x, right_y), 5, (255, 0, 0), -5)
                face_frame = cv2.circle(
                    face_frame, (left_x, left_y), 5, (255, 0, 0), -5)
                cv2.putText(face_frame, "yaw:{:.2f} - pitch:{:.2f} - roll:{:.2f}".format(
                    yaw, pitch, roll), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                cv2.putText(face_frame, "gaze-vector x:{:.2f} - y:{:.2f} - z:{:.2f}".format(
                    yaw, pitch, roll), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                cv2.imshow('left eye', left_eye_image)
                cv2.imshow('right eye', right_eye_image)
                x, y, z = gaze_vector
                cv2.arrowedLine(face_frame, (left_x, left_y), (left_x +
                                                               int(x*100), left_y + int(-y*100)), (0, 0, 255), 2)
                cv2.arrowedLine(face_frame, (right_x, right_y), (right_x +
                                                                 int(x*100), right_y + int(-y*100)), (0, 0, 255), 2)
                cv2.imshow('face detection', face_frame)
                cv2.waitKey(60)

        else:
            break
    # inference benchmarks

    logging.info(f"------Inference Times {args.precision}------")
    logging.info("Face Detection: {:.5f} sec".format(inf_time[0]/inf_time[4]))
    logging.info("Facial Landmarks: {:.5f} sec".format(
        inf_time[1]/inf_time[4]))
    logging.info(
        "Head Pose Estimation: {:.5f} sec".format(inf_time[2]/inf_time[4]))
    logging.info("Gaze Estimation: {:.5f} sec".format(inf_time[3]/inf_time[4]))
    input_feed.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler("app.log")
        ])
    args = get_args()
    main(args)
