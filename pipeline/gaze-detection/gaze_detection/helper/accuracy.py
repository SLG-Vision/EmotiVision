from components.face_detection import HaarCascade
import cv2
import dlib
from pathlib import Path
import time
import os

"""
    Script to test how good the haarCascade OpenCv detector works using the Fer2013 dataset.
    With my current settings, the results are following. 
    {'0': 15272, '1': 20608, '2': 7}
    Please note that the value of each key represents the number of images in which that key number of faces have been detected.
    Using the dlib hog based face detector we achive the following results:
    {'0': 10932, '1': 24955, '2': 0}
"""
FER2013_PATH = "fer2013"

algorithms = ["hog_dlib", "haar_opencv"]
haarCascade = HaarCascade()
face_haar_cascade = cv2.CascadeClassifier(os.path.join('predictors','haarcascade_frontalface_alt.xml'))
face_hog_detector = dlib.get_frontal_face_detector()
face_detected = {'0':0, '1':0, '2':0}

def print_results(time_start, r=True):
    endl="\r" if r else "\n"
    print("Results: {} - Time Elapsed: {:.2f}".format(face_detected, time.time()-time_start), end=endl)

def detect_faces(algorithm, path=FER2013_PATH):
    time_start = time.time()
    for image_path in list(Path(path).rglob("*.jpg")):
        img = cv2.imread(str(image_path))
        
        if algorithm==algorithms[0]: #hog_dlib
            faces = face_hog_detector(img, 1)
        elif algorithm==algorithms[1]: #haar_opencv
            scale_factor=1.1
            min_neighbors=2
            faces = face_haar_cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scale_factor, min_neighbors)
            
        
        face_detected[str(len(faces))]+=1
        print_results(time_start)
        
    print_results(time_start, False)          
    
    
for algorithm in algorithms:
    face_detected = {'0':0, '1':0, '2':0}
    print(f"\n# FACES DETECTED PER IMAGE USING {algorithm}:")
    detect_faces(algorithm)

