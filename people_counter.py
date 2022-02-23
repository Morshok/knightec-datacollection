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
argument_parser.add_argument("-s", "--skip-frames", type=int, default=30,
    help="Number of frames skipped between detections");
args = vars(argument_parser.parse_args());

print("[INFO]: Starting video stream from Webcam...");
videoStream = cv2.VideoCapture(0, cv2.CAP_V4L2);
time.sleep(2.0);

(Height, Width) = (None, None);
maxWidth = 1920;

centroid_tracker = CentroidTracker(30);
trackers = [];
trackable_objects = { };

cascadePath = "./haarcascade_frontalface_default.xml";
faceCascade = cv2.CascadeClassifier(cascadePath);

totalFrames = 0;
totalUp = 0;
totalDown = 0;

fps = FPS().start();

while True:
    check, frame = videoStream.read();

    frame = imutils.resize(frame, width=maxWidth);
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB);

    if Width is None or Height is None:
        (Width, Height) = frame.shape[:2];
    
    status = "Waiting";
    rects = [];

    # Check if we should use object detection
    if totalFrames % args["skip_frames"] == 0:
        status = "Detecting";
        trackers = [];

        faces = faceCascade.detectMultiScale(
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
        trackable_object = TrackableObject(objectID, centroid);

        #trackable_object = objects.get(objectID, None);

        #print(centroid)
        #print(trackable_object)        
        if trackable_object is None:
            trackable_object = TrackableObject(objectID, centroid);
        else:
            y = [c[1] for c in trackable_object.centroids];
            direction = centroid[1] - np.mean(y);
            trackable_object.centroids.append(centroid);
        
            if not trackable_object.hasBeenCounted:
                # Check if object is moving up, and if it is above
                # the horizontal line in the middle. Count it and 
                # set it as counted if that is the case
                if direction < 0 and centroid[1] < Height // 2:
                    totalUp += 1;
                    trackable_object.hasBeenCounted = True;
                # Check if object is moving down, and if it is below
                # the horizontal line in the middle. Count it and 
                # set it as counted if that is the case
                elif direction > 0 and centroid[1] > Height // 2:
                    totalDown += 1;
                    trackable_object.hasBeenCounted = True;
        # Store the trackable object in our dictionary
        trackable_objects[objectID] = trackable_object;

        # Draw both ID of the object and its centroid
        text = "ID {}".format(objectID);
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2);
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1);
    
    # Construct tuple of information to display on the frame
    info = [ ("Up", totalUp), ("Down", totalDown), ("Status", status) ];

    # Loop through the info tuples and display them on the frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v);
        cv2.putText(frame, text, (10, Height - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2);
    
    # Display the frame
    cv2.imshow("Frame", frame);

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break;
    
    # Update timer and total number of frames elapsed
    totalFrames += 1;
    totalFrames %= args["skip_frames"];
    fps.update();

# Print information
fps.stop();
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()));
print("[INFO] approximate FPS: {:.2f}".format(fps.fps()));

# Release resources and close application
videoStream.release();
cv2.destroyAllWindows();