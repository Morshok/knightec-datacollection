from tracking.centroidtracker import CentroidTracker
from tracking.trackableobject import TrackableObject
from tracking.trackableobject import Direction
from imutils.video import FPS
from aws_handler import *
import numpy as np
import threading
import argparse
import imutils
import time
import dlib
import time
import cv2
import os

# Argument parser if code is run from terminal, with set default 
# values in case code is run inside of an IDE, or no arguments 
# are given from terminal
argument_parser = argparse.ArgumentParser();
argument_parser.add_argument("-y", "--yolo", default="yolov5", 
                            help="Base path to yolo directory");
argument_parser.add_argument("-c", "--confidence", type=float, default=0.5, 
                            help="Minimum probability to filter out weak detections");
argument_parser.add_argument("-t", "--threshold", type=float, default=0.3, 
                            help="Threshold when applying non maxima suppression");
argument_parser.add_argument("-s", "--skip-seconds", type=int, default=1, 
                            help="Number of seconds skipped between detections");
args = vars(argument_parser.parse_args());

def load_video_capture(use_camera):
    print("[INFO]: Starting video stream...");

    if(not use_camera):
        video_capture = cv2.VideoCapture("./video/example_01.mp4");
        video_framerate = video_capture.get(cv2.CAP_PROP_FPS);
        return video_capture, video_framerate;

    video_capture = cv2.VideoCapture(0, cv2.CAP_V4L2);
    return video_capture, None;

def load_coco_class_labels(args):
    classesPath = os.path.sep.join([args["yolo"], "coco.names"]);
    classes = open(classesPath).read().strip().split("\n");

    return classes;

def build_yolov5_net(args):
    print("[INFO]: loading YoloV5 from disk...");

    onnxPath = os.path.sep.join([args["yolo"], "yolov5s.onnx"]);

    net = cv2.dnn.readNet(onnxPath);
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16);

    return net;

# Load video capture and get video frame rate.
# Set use_camera to True to use camera instead of video
video_capture, video_framerate = load_video_capture(use_camera=False);
video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1);

# Load the COCO class labels our YOLO model was trained on
CLASSES = load_coco_class_labels(args=args);

# Load our YOLO object detector trained on COCO dataset
# given the path to the onnx file
net = build_yolov5_net(args=args);

# Variables for getting the height 
# and the width of the frame
(Height, Width) = (None, None);

# Variables for controlling the width and height
# while performing OpenCV operations
processing_width = 300;
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

# Variables for how long we should wait
# until we contact aws, and whether or not
# an update is already in progress
aws_update_time_sec = 15;
aws_updating_in_progress = False;

def create_AWS_thread():
    global aws_update_time_sec, aws_updating_in_progress;

    if not aws_updating_in_progress:
        print(f"[INFO]: Contacting AWS for updates in {aws_update_time_sec} seconds");
        thread = threading.Thread(target=contact_AWS);
        thread.start();
        aws_updating_in_progress = True;

def contact_AWS():
    global aws_update_time_sec, aws_updating_in_progress, total_in, total_out;

    time.sleep(aws_update_time_sec);

    print("[Info]: Contacting Aws...");
    add_entry_event(total_in, total_out);
    print(f"{total_in} persons entered");
    print(f"{total_out} persons exited");
    total_in = 0;
    total_out = 0;
    aws_updating_in_progress = False;

def format_for_yolov5_inference(frame):
    ''' 
        Method for formatting an input frame for
        Use in YOLOv5 inference. YOLOv5 accepts an
        image of size 300x300.
    '''

    # Get width and height from the input frame
    # and determine which of the two has the 
    # highest value
    width, height, channels = frame.shape;
    maxOfWidthOrHeight = max(width, height);

    # Create a rectangle that is maxOfWidthOrHeight x maxOfWidthOrHeight
    # and set all pixel values to zero. Then put the original image 
    # back between x=(0-width) and y=(0-height)
    result = np.zeros((maxOfWidthOrHeight, maxOfWidthOrHeight, 3), np.uint8);
    result[0:width, 0:height] = frame;
    
    return result;

def perform_yolov5_inference(input_image, net):
    ''' 
        Method for performing YOLOv5 inference. The input
        image is converted into a blob that is later input
        into the YOLOv5 network for inference. The inference
        time is measured and printed, and the method returns
        the predictions from the YOLOv5 inference.
    '''

    # Convert our input image into a blob and set it as
    # input to our YOLOv5 network.
    blob = cv2.dnn.blobFromImage(input_image, 1 / 255.0, (640, 640), swapRB=True, crop=False);
    net.setInput(blob);
    
    # Perform YOLOv5 inference with input blob
    # and measure inference time
    before = time.perf_counter();
    predictions = net.forward();
    after = time.perf_counter();
    print(f"[INFO]: YoloV5 inference time: {after - before}s");

    return predictions;

def unwrap_yolov5_detections(input_image, predictions, confidence_threshold, nms_threshold):
    ''' 
        Method for unwrapping the predictions from the
        YOLOv5 inference. Checks whether class confidence
        is above set confidence threshold and performs
        Non-Maxima Suppression.
    '''
    # Create empty lists for resulting bounding boxes,
    # corresponding classId's and confidences
    boxes = [];
    classIDs = [];
    confidences = [];

    # Get total amount of predictions
    num_rows = predictions.shape[0];

    # Get input image dimensions and initiate
    # two scale factor variables
    image_width, image_height, channels = input_image.shape;
    x_factor = image_width / 640;
    y_factor = image_height / 640;

    # Loop through all predictions row by row
    for r in range(num_rows):
        # Get prediction at row index r and extract its confidence score
        row = predictions[r];
        confidence = row[4];

        # Check if the confidence score is greater than the 
        # confidence threshold. This filters out weak detections.
        if confidence > confidence_threshold:
            # Extract the classID of the detection
            class_scores = row[5:];
            _, _, _, max_index = cv2.minMaxLoc(class_scores);
            classID = max_index[1];

            # Extract the bounding box coordinates of the detection and 
            # construct a box given said coordinates
            (x, y, w, h) = (row[0].item(), row[1].item(), row[2].item(), row[3].item())
            (startX, startY, endX, endY) = (int((x - 0.5 * w) * x_factor), int((y - 0.5 * h) * y_factor), int(w * x_factor), int(h * y_factor));
            box = np.array([startX, startY, endX, endY]);

            # Add variables to our lists
            boxes.append(box);
            classIDs.append(classID);
            confidences.append(confidence);

    # Perform non-maxima suppression, which removes overlapping
    # bounding boxes representing the same object, and get the 
    # resulting indices in our lists.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold);

    # Get the boxes, classID's and confidences
    # after non-maxima suppression has been performed
    # using the indices given by the function.
    resulting_boxes = [];
    resulting_classIDs = [];
    resulting_confidences = [];

    for i in indices:
        resulting_boxes.append(boxes[i]);
        resulting_classIDs.append(classIDs[i]);
        resulting_confidences.append(confidences[i]);

    return resulting_boxes, resulting_classIDs, resulting_confidences;

def process_yolov5_detection(boxes, classIDs, confidences, rgb_frame):
    ''' 
        Method for going through our unwrapped predictions
        and checking if the detected object class corresponds
        to that of a person. Then bounding box coordinates
        are extracted and used to create a new kernalized 
        correlation filter object tracker and add it to our
        list of object trackers. 
    '''
    global CLASSES, trackers;

    for (box, classID, confidence) in zip(boxes, classIDs, confidences):
        if CLASSES[classID] != "person":
            continue;
        
        # Extract bounding box coordinates
        (x, y) = (box[0], box[1]);
        (w, h) = (box[2], box[3]);
        (startX, startY, endX, endY) = (x, y, x + w, y + h);

        # Construct a dlib rectangle from bounding 
        # box coordinates and start a dlib 
        # correlation tracker. 
        tracker = dlib.correlation_tracker();
        rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY));
        tracker.start_track(rgb_frame, rect);

        # Add the tracker to our list of trackers
        # so we can utilize it during skip frames
        trackers.append(tracker);

# Get start time
start_time = time.perf_counter();
while True:
    # Get current time
    current_time = time.perf_counter();

    # Read frame from Videocapture
    check, frame = video_capture.read();

    # If no more frames could be read
    # (End of video), break out of the 
    # loop, terminating the program
    if frame is None:
        break;

    # Resize frame and convert from BGR to RGB
    frame = imutils.resize(frame, width=processing_width);
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB);

    # Get width of the frame if not already fetched
    if Width is None or Height is None:
        (Height, Width) = frame.shape[:2];
    
    status = "Waiting";
    rects = [];

    # Check if we should use object detection
    if (current_time - start_time) >= args["skip_seconds"]:
        start_time = current_time;
        status = "Detecting";
        trackers = [];

        input_image = format_for_yolov5_inference(frame);
        predictions = perform_yolov5_inference(input_image, net);
        boxes, classIDs, confidences = unwrap_yolov5_detections(input_image, predictions[0], args["confidence"], args["threshold"]);
        process_yolov5_detection(boxes, classIDs, confidences, rgb_frame);

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

    # Loop through all old objects from the centroid tracker
    for (objectID, centroid) in objects.items():

        # Try and get trackable object with objectID
        trackable_object = trackable_objects.get(objectID, None);
            
        # If trackable object is None, create a 
        # new trackable object with objectID
        if trackable_object is None:
            trackable_object = TrackableObject(objectID, centroid);
        # Else, determine the direction of movement for the current
        # centroid, and increment total_in/total_out as well as 
        # contact aws if applicable.
        else:
            y = [c[1] for c in trackable_object.centroids];
            direction = centroid[1] - np.mean(y);
            trackable_object.centroids.append(centroid);

            if not (trackable_object.trackingDirection == Direction.Down) and direction < 0 and centroid[1] < Height // 2:
                create_AWS_thread();
                total_out += 1;
                trackable_object.trackingDirection = Direction.Down;
            elif not (trackable_object.trackingDirection == Direction.Up) and direction > 0 and centroid[1] > Height // 2:
                create_AWS_thread();
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

    # Check if 'q' has been pressed, and 
    # break out of the loop, terminating
    # the program if it has been
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
print("[INFO]: Terminating...")