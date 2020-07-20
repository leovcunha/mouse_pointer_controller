
import cv2
import os
from argparse import ArgumentParser
import numpy as np
import logging
import time
from input_feeder import InputFeeder
from mouse_controller import MouseController
from face_detection import Face_Detection


def get_args():
    parser = ArgumentParser(
        description='Mouse Pointer Controller using eye gaze')
    parser.add_argument("-t", "--input-type", required=True, type=str,
                        help="Type of input (video or cam)")
    parser.add_argument("-i", "--input", required=True, type=str,
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

    input_feed = InputFeeder(args.input_type, args.input)
    input_feed.load_data()

    for frame in input_feed.next_batch():
        if frame is not None:
            show_frame = fd.predict(frame.copy())
            cv2.imshow('face detection', show_frame)
            cv2.waitKey(0)
        else:
            break

    input_feed.close()


if __name__ == '__main__':
    args = get_args()
    main(args)
