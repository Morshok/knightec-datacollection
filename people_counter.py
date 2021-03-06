from centroidtracker import CentroidTracker
from trackableobject import TrackableObject
from trackableobject import Direction
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

print("[INFO]: Starting video stream from Webcam...");
video_Stream = cv2.VideoCapture(0, cv2.CAP_V4L2);
video_Stream.set(cv2.CAP_PROP_BUFFERSIZE, 1);

(Height, Width) = (None, None);
processing_width = 250;
output_width = 700;

centroid_tracker = CentroidTracker(30);
trackers = [];
trackable_objects = { };

cascade_path = "./haarcascade_frontalface_default.xml";
face_cascade = cv2.CascadeClassifier(cascade_path);

total_frames = 0;
total_up = 0;
total_down = 0;

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
    global aws_update_time_sec, aws_updating_in_progress, total_down, total_up;

    time.sleep(aws_update_time_sec);

    print("[Info] Contacting Aws...");
    time.sleep(2.0);
    print("{} persons entered".format(total_down));
    print("{} persons exited".format(total_up));
    total_down = 0;
    total_up = 0;
    aws_updating_in_progress = False;

start_time = time.time();
while True:
    current_time = time.time();

    check, frame = video_Stream.read();

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

        faces = face_cascade.detectMultiScale(
            frame,
            scaleFactor = 1.1,
            minNeighbors = 5
        );

        for x, y, w, h in faces:
            (startX, startY, endX, endY) = (x, y, x+w, y+h);
            
            tracker = dlib.correlation_tracker();
            rect = dlib.rectangle(startX, startY, endX, endY);
            tracker.start_track(rgb_frame, rect);

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
                total_up += 1;
                trackable_object.trackingDirection = Direction.Down;
            elif not (trackable_object.trackingDirection == Direction.Up) and direction > 0 and centroid[1] > Height // 2:
                create_AWS_thread();
                total_down += 1;
                trackable_object.trackingDirection = Direction.Up;

        # Store the trackable object in our dictionary
        trackable_objects[objectID] = trackable_object;

        # Draw both ID of the object and its centroid
        text = "ID {}".format(objectID);
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2);
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1);
    
    # Construct tuple of information to display on the frame
    info = [ ("Up", total_up), ("Down", total_down), ("Status", status) ];

    # Loop through the info tuples and display them on the frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v);
        cv2.putText(frame, text, (10,  ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2);
    
    # Display the frame
    frame = imutils.resize(frame, width=output_width);
    cv2.imshow("Frame", frame);

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break;
    
    # Update timer and total number of frames elapsed
    fps.update();

# Print information
fps.stop();
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()));
print("[INFO] approximate FPS: {:.2f}".format(fps.fps()));

# Release resources and close application
video_Stream.release();
cv2.destroyAllWindows();