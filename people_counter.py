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
import os

argument_parser = argparse.ArgumentParser();
argument_parser.add_argument("-y", "--yolo", default="yolov5", help="Base path to yolo directory");
argument_parser.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum probability to filter out weak detections");
argument_parser.add_argument("-t", "--threshold", type=float, default=0.3, help="Threshold when applying non maxima suppression");
argument_parser.add_argument("-s", "--skip-seconds", type=int, default=1, help="Number of frames skipped between detections");
args = vars(argument_parser.parse_args());

#print("[INFO]: Starting video stream from Webcam...");
video_Stream = cv2.VideoCapture("./video/example_01.mp4");
video_Stream.set(cv2.CAP_PROP_BUFFERSIZE, 1);
video_framerate = video_Stream.get(cv2.CAP_PROP_FPS) + 10;

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"]);
CLASSES = open(labelsPath).read().strip().split("\n");

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov5s.weights"]);
configurationPath = os.path.sep.join([args["yolo"], "yolov5s.cfg"]);

# load our YOLO object detector trained on COCO dataset
print("[INFO] loading YOLO from disk...");
net = cv2.dnn.readNetFromDarknet(configurationPath, weightsPath);

# determine only the *output* layer names that we need from YOLO
outputLayers = net.getUnconnectedOutLayersNames();

(Height, Width) = (None, None);
processing_width = 250;
output_width = 700;

centroid_tracker = CentroidTracker(30);
trackers = [];
trackable_objects = { };

total_out = 0;
total_in = 0;

fps = FPS().start();

# After how many seconds should we contanct AWS
# and reset our variables?
aws_update_time_sec = 15;
aws_updating_in_progress = False;

def create_AWS_thread():
    global aws_updating_in_progress;

    if not aws_updating_in_progress:
        print("[INFO] Contacting AWS for updates in {} seconds".format(aws_update_time_sec));
        thread = threading.Thread(target=contact_AWS);
        thread.start();
        aws_updating_in_progress = True;

def contact_AWS():
    global aws_update_time_sec, aws_updating_in_progress, total_in, total_out;

    time.sleep(aws_update_time_sec);

    print("[Info] Contacting Aws...");
    time.sleep(2.0);
    print("{} persons entered".format(total_in));
    print("{} persons exited".format(total_out));
    total_in = 0;
    total_out = 0;
    aws_updating_in_progress = False;

start_time = time.time();
while True:
    current_time = time.time();

    check, frame = video_Stream.read();

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
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False);
        net.setInput(blob);
        layerOutputs = net.forward(outputLayers);

        # Initialize our lists of bounding boxes, confidences and classIDs respectively
        boxes = [];
        confidences = [];
        classIDs = [];

        # Loop through each layer output
        for output in layerOutputs:
            # Loop through each detection of each output
            for detection in output:
                # Extract classID and confidence score of current detection
                scores = detection[5:];
                classID = np.argmax(scores);
                confidence = scores[classID];
                
                # Filter out weak detections
                if confidence > args["confidence"]:
                    # Scale bounding box coordinates back to size of original image
                    box = detection[0:4] * np.array([Width, Height, Width, Height]);
                    (centerX, centerY, width, height) = box.astype("int");

                    # Use (x, y)-coordinates to derive top and left corner of bounding box
                    x = int(centerX - (width / 2));
                    y = int(centerY - (height / 2));

                    # Update our lists of boxes, confidences and classIDs
                    boxes.append([x, y, int(width), int(height)]);
                    confidences.append(float(confidence));
                    classIDs.append(classID);
        
        # Apply non maxima suppression to suppress weak, overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"]);

        # Ensure that at least one detection exists
        if len(indices) > 0:
            # Loop over the indices we've kept
            for i in indices.flatten():
                # Check whether detection is a person or not
                if CLASSES[classIDs[i]] != 'person':
                    continue;
                
                # Extract bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1]);
                (w, h) = (boxes[i][2], boxes[i][3]);
                (startX, startY, endX, endY) = (x, y, x + w, y + h);

                # Construct a dlib rectangle from bounding 
                # box coordinates and start a dlib 
                # correlation tracker. 
                tracker = dlib.correlation_tracker();
                rect = dlib.rectangle(startX, startY, endX, endY);
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
                create_AWS_thread();
                total_out += 1;
                trackable_object.trackingDirection = Direction.Down;
            elif not (trackable_object.trackingDirection == Direction.Up) and direction > 0 and centroid[1] > Height // 2:
                create_AWS_thread();
                total_in += 1;
                trackable_object.trackingDirection = Direction.Up;

        # Store the trackable object in our dictionary
        trackable_objects[objectID] = trackable_object;

        # Draw both ID of the object and its centroid
        text = "ID {}".format(objectID);
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2);
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1);
    
    # Construct tuple of information to display on the frame
    info = [ ("Exited", total_out), ("Entered", total_in), ("Status", status) ];

    # Loop through the info tuples and display them on the frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v);
        cv2.putText(frame, text, (10,  ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2);
    
    # Display the frame
    frame = imutils.resize(frame, width=output_width);
    cv2.imshow("Frame", frame);

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break;
    
    # Update timer 
    fps.update();

    # Ensure fixed framerate based on video framerate
    timeDiff = time.time() - current_time;
    if(timeDiff < 1.0/(video_framerate)):
        time.sleep(1.0/(video_framerate) - timeDiff);

# Print information
fps.stop();
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()));
print("[INFO] approximate FPS: {:.2f}".format(fps.fps()));

# Release resources and close application
video_Stream.release();
cv2.destroyAllWindows();