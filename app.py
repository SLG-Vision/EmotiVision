from gazedetection.gaze_detection.main import GazeDetection
from face_retrieval.src.face_retrieval.retrieval import Retrieval
import cv2


gaze_detection = GazeDetection(predictor_path="shape_predictor_68_face_landmarks.dat", video=False, print_on_serial=False, crop_frame_paddings=(0.3,0,0,0.1))
retrieval = Retrieval("def_blacklist.pt")

vid = cv2.VideoCapture(0)
while(True):
    _, frame = vid.read()

    return_frame, is_gaze_facing = gaze_detection.detect(frame)
    
    ret = retrieval.evaluateFrame(return_frame)
    
    print(f"GAZE IS FACING CAMERA:     {is_gaze_facing}       \nIs blacklisted:     {'Found' if ret else 'Not Found'}", end='\r')

    cv2.imshow('frame',  return_frame)
                
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()


