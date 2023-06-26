from face_retrieval.src.retrieval import Retrieval
from gaze_detection.src.main import GazeDetection
from fer_classification.net.FERModule import FER
import cv2

net = FER()
net.load()

gaze_detection = GazeDetection(predictor_path="shape_predictor_68_face_landmarks.dat", video=False, print_on_serial=False, crop_frame_paddings=(0.0,0.0,0.0,0.0))

# performing hyperparameters: {thr: 0:18, usingAverage=True}
retrieval = Retrieval("me_noaugmentation_blacklist.pt", threshold=0.18, debug=True, distanceMetric='cosine', usingAverage=True, usingMedian=False, usingMax=False, toVisualize=True, usingMtcnn=False)

vid = cv2.VideoCapture(0)
while(True):
    _, frame = vid.read()

    return_frame, is_gaze_facing = gaze_detection.detect(frame)
    if(is_gaze_facing == False):
        pass    # no tracking condition
    
    # recall peggiore:
    code, retrieval_return_string = retrieval.evalFrameTextual(return_frame)

    # recall molto migliore
    
    #code, retrieval_return_string = retrieval.evalFrameTextual(frame)

    # recall massima
    
    #retrieval.setUsingMtcnn(True)
    #code, retrieval_return_string = retrieval.evalFrameTextual(frame)
    
    if(retrieval.hasMtcnnFailed()):
        continue
    
    if(retrieval.isPersonBlacklisted()):
        pass    # no tracking condition
    
    print(f"\nGaze is facing:\t{is_gaze_facing}\t\n\tRetrieval status: {retrieval_return_string}\t\n\tEmotion Detected: {net.predict(return_frame)}")
    cv2.imshow('frame',  return_frame)
                
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
