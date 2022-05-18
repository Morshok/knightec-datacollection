from tracking.centroidtracker import CentroidTracker
from tracking.trackableobject import TrackableObject
from tracking.trackableobject import Direction
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import threading
import argparse
import imutils
import time
import dlib
import time
import cv2

argument_parser = argparse.ArgumentParser();
argument_parser.add_argument("-s", "--skip-seconds", type=int, default=1,
    help="Number of frames skipped between detections");
args = vars(argument_parser.parse_args());

def load_video_capture(use_camera):
    print("[INFO]: Starting video stream...");

    if(not use_camera):
        video_capture = cv2.VideoCapture("./video/example_01.mp4");
        video_framerate = video_capture.get(cv2.CAP_PROP_FPS) + 10;
        return video_capture, video_framerate;

    video_capture = cv2.VideoCapture(0, cv2.CAP_V4L2);
    return video_capture, None;

def get_class_labels(args):
    classes = [
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"
    ];

    return classes;

def build_MobileNetSSD_net():
    print("[INFO]: loading model...");

    net = cv2.dnn.readNetFromCaffe("./mobilenet_ssd/MobileNetSSD_deploy.prototxt", "./mobilenet_ssd/MobileNetSSD_deploy.caffemodel");
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA);

    return net;

# Load video capture
video_capture, video_framerate = load_video_capture(use_camera=False);
video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1);

# load our MobileNet-SSD model from disk
net = build_MobileNetSSD_net();

# Threshold to filter out weak detections
confidenceThreshold = 0.5;

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = get_class_labels();

# Variables for getting the height 
# and the width of the frame
(Height, Width) = (None, None);

# Variables for controlling the width and height
# while performing OpenCV operations
processing_width = 250;
output_width = 700;

# Initialize a centroid tracker with
# a maxDisappeared value of 30
centroid_tracker = CentroidTracker(30);

# Initialize lists and arrays for 
# trackers and trackable objects
trackers = [];
trackable_objects = { };

# Variables for keeping track of 
# how many people have moved in or out
(total_out, total_in) = (0, 0);

# Initialize and start fps counter
fps = FPS().start();

# After how many seconds should we contanct AWS
# and reset our variables?
aws_update_time_sec = 15;
aws_updating_in_progress = False;

def create_AWS_thread():
    global aws_updating_in_progress;

    if not aws_updating_in_progress:
        print("[INFO]: Contacting AWS for updates in {} seconds".format(aws_update_time_sec));
        thread = threading.Thread(target=contact_AWS);
        thread.start();
        aws_updating_in_progress = True;

def contact_AWS():
    global aws_update_time_sec, aws_updating_in_progress, total_in, total_out;

    time.sleep(aws_update_time_sec);

    print("[Info]: Contacting Aws...");
    time.sleep(2.0);
    print("{} persons entered".format(total_in));
    print("{} persons exited".format(total_out));
    total_in = 0;
    total_out = 0;
    aws_updating_in_progress = False;

start_time = time.time();
while True:
    current_time = time.time();

    check, frame = video_capture.read();

    if frame is None:
        break;

    frame = imutils.resize(frame, width=processing_width);
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB);

    if Width is None or Height is None:
        (Height, Width) = frame.shape[:2];
    
    status = "Waiting";
    rects = [];

    # Check if we should use object detection
    if (current_time - start_time) >= args["skip_seconds"]:
        start_time = current_time;
        status = "Detecting";
        trackers = [];

        # Convert frame into blob and pass it
        # through the network to obtain the detections
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (Width, Height), 127.5);
        before = time.perf_counter();
        net.setInput(blob);
        detections = net.forward();
        after = time.perf_counter();
        print(f'Inference time: {after - before}s');

        # Loop through the detections
        for i in np.arange(0, detections.shape[2]):
            # Extract the confidence of the prediction
            # from the detection
            confidence = detections[0, 0, i, 2];

            # Filter out the weak detections by 
            # requiring a minumum confidence
            if confidence > confidenceThreshold:
                # Extract the index of the class 
                # label and check if it is person
                index = int(detections[0, 0, i, 1]);

                if CLASSES[index] != 'person':
                    continue;
                
                # Compute the (x, y) - coordinates for the 
                # bounding box of the detected person
                box = detections[0, 0, i, 3:7] * np.array([Width, Height, Width, Height]);
                (startX, startY, endX, endY) = box.astype("int");

                # Construct a dlib rectangle from bounding 
                # box coordinates and start a dlib 
                # correlation tracker. 
                tracker = dlib.correlation_tracker();
                rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY));
                tracker.start_track(rgb_frame, rect);

                # Add the tracker to our list of trackers
                # so we can utilize it during skip frames
                trackers.append(tracker);
    # If not, utilize object trackers to decrease computational cost
    else:
        for tracker in trackers:
            status = "Tracking";

            tracker.update(rgb_frame);

            position = tracker.get_position();
            (startX, startY, endX, endY) = (int(position.left()), int(position.top()), int(position.right()), int(position.bottom()));
            rects.append((startX, startY, endX, endY));
    
    # Draw horizontal line in the middle of the screen
    cv2.line(frame, (0, Height // 2), (Width, Height // 2), (0, 255, 255), 2);
    
    # Get old objects from centroid tracker
    objects = centroid_tracker.update(rects);

    for (objectID, centroid) in objects.items():

        trackable_object = trackable_objects.get(objectID, None);
            
        if trackable_object is None:
            trackable_object = TrackableObject(objectID, centroid);
        else:
            y = [c[1] for c in trackable_object.centroids];
            direction = centroid[1] - np.mean(y);
            trackable_object.centroids.append(centroid);

            if not (trackable_object.trackingDirection == Direction.Down) and direction < 0 and centroid[1] < Height // 2:
                # create_AWS_thread();
                total_out += 1;
                trackable_object.trackingDirection = Direction.Down;
            elif not (trackable_object.trackingDirection == Direction.Up) and direction > 0 and centroid[1] > Height // 2:
                # create_AWS_thread();
                total_in += 1;
                trackable_object.trackingDirection = Direction.Up;

        # Store the trackable object in our dictionary
        trackable_objects[objectID] = trackable_object;

        # Display ID of the object and draw its centroid
        cv2.putText(frame, f"ID {objectID}", (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2);
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1);
    
    # Display interesting information on the frame,
    # such as how many people have entered, exited,
    # and in what state the program is currently in 
    # (waiting, detecting, tracking)
    cv2.putText(frame, f"Exited: {total_out}", (10,  20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2);
    cv2.putText(frame, f"Entered: {total_in}", (10,  40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2);
    cv2.putText(frame, f"Status: {status}", (10,  60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2);
    
    # Display the frame
    frame = imutils.resize(frame, width=output_width);
    cv2.imshow("Frame", frame);

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break;
    
    # Update timer 
    fps.update();

    # Check if pre-recorded video is used
    # and ensure fixed framerate based on 
    # the variable video_framerate
    if not video_framerate == None:
        timeDiff = time.time() - current_time;
        if(timeDiff < 1.0/(video_framerate)):
            time.sleep(1.0/(video_framerate) - timeDiff);

# Stop fps counter
fps.stop();

# Print information
print(f"[INFO]: elapsed time: {fps.elapsed():.2f}");
print(f"[INFO]: approximate FPS: {fps.fps():.2f}");
print(f'[INFO]: Found {total_in} people entering, and {total_out} people exiting...');

# Release resources and close application
video_capture.release();
cv2.destroyAllWindows();

print("[INFO]: Terminating...");