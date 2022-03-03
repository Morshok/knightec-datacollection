from sys import flags
import cv2;
from imutils.video import FPS

#flip=2;
#dispW=320;
#dispH=240;
#nativeW=640
#nativeH=480
#camSet = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width='+str(nativeW)+', height='+str(nativeH)+', format=NV12, framerate=21/1 ! nvvidconv flip-methods='+str(flip)+'! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink';

video = cv2.VideoCapture(0, cv2.CAP_V4L2);
video.set(cv2.CAP_PROP_BUFFERSIZE, 0);
if not (video.isOpened()):
    print ("Not open")

cascadePath = "./haarcascade_frontalface_default.xml";
faceCascade = cv2.CascadeClassifier(cascadePath);

fps = FPS().start();

while True:
    check, frame = video.read();

    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor = 1.1,
        minNeighbors = 5
    );

    for x, y, w, h in faces:
        frame = cv2.rectangle(
            frame, 
            (x, y), 
            (x + w, y + h), 
            (0, 255, 0), 
            3
        );
           
    cv2.imshow('Face Detector', frame);

    key = cv2.waitKey(1);
    if key == ord('q'):
        break;
    
    fps.update();

fps.stop();
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()));
print("[INFO] approximate FPS: {:.2f}".format(fps.fps()));

video.release();
cv2.destroyAllWindows();