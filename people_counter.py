from centroidtracker import CentroidTracker
from trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

argument_parser = argparse.ArgumentParser();
argument_parser.add_argument("-c", "--confidence", type=float, default=0.4,
    help="Minimum probability to filter out weak detections");
argument_parser.add_argument("-s", "--skip-frames", type=int, default=30,
    help="Number of frames skipped between detections");

print("[INFO]: Starting video stream from Webcam...");
videoStream = VideoStream(src=0).start();
time.sleep(2.0);

(Height, Width) = (None, None);