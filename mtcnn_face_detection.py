from turtle import color
import cv2
from mtcnn import MTCNN

kernelSize = (101, 101);
font = cv2.FONT_HERSHEY_SIMPLEX;

def mtcnn_find_face(frame, result_list):
    for result in result_list:
        x, y, w, h = result['box'];
        roi = frame[y:y+h, x:x+w];
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 155, 255), 5);
        detectedFace = cv2.GaussianBlur(roi, kernelSize, 0);
        frame[y:y+h, x:x+w] = detectedFace;
    return frame;

videoCapture = cv2.VideoCapture(0, cv2.CAP_DSHOW);
detector = MTCNN();

while True:
    check, frame = videoCapture.read();

    if not check:
        print("Could not read video input from Webcam");
        break;

    faces = detector.detect_faces(frame);
    detectFaceMTCNN = mtcnn_find_face(frame, faces);

    cv2.imshow("Video", detectFaceMTCNN);

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break;

videoCapture.release();
cv2.destroyAllWindows();