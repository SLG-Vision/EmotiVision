from face_retrieval.src.retrieval import Retrieval
from gaze_detection.gaze_detection.main import GazeDetection
from fer_classification.net.FERNet import FERNet
import cv2
import torch

import os
print(os.path.abspath('shape_predictor_68_face_landmarks.dat'))

gaze_detection = GazeDetection(predictor_path="shape_predictor_68_face_landmarks.dat", video=False, print_on_serial=False, crop_frame_paddings=(0.3,0,0,0.1))
retrieval = Retrieval("def_blacklist.pt")

net = FERNet()
net.load_state_dict(torch.load('pipeline/fer_classification/student_distilled.t7'))
label = {0:'angry', 1:'disgust', 2:'fear', 3:'happy',  4:'sad', 5:'surprise', 6:'neutral'}

vid = cv2.VideoCapture(0)
while(True):
    _, frame = vid.read()

    return_frame, is_gaze_facing = gaze_detection.detect(frame)
    pred = net(return_frame)
    ret = retrieval.evaluateFrame(return_frame)
    
    print(f"Gaze is facing:     {is_gaze_facing} \n Is blacklisted:     {'Found' if ret else 'Not Found'} \n Emotion Detected: {label[return_frame]}")

    cv2.imshow('frame',  return_frame)
                
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()