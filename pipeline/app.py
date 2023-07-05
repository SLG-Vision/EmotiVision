from face_retrieval.src.retrieval import Retrieval
from gaze_detection.src.main import GazeDetection
from fer_classification.net.FERModule import FER
import cv2
import time
net = FER()
net.load()

gaze_detection = GazeDetection(predictor_path="shape_predictor_68_face_landmarks.dat", video=False, pupil_detection_mode="filtering", print_on_serial=True, crop_frame_paddings=(0.0,0.0,0.0,0.0), serial_port="/dev/tty.usbmodem11401")

# performing hyperparameters: {thr: 0:18, usingAverage=True}
retrieval = Retrieval("Zuckerberg_blacklist.pt", threshold=0.18, debug=True, distanceMetric='cosine', usingAverage=True, usingMedian=False, usingMax=False, toVisualize=False, usingMtcnn=False)

vid = cv2.VideoCapture(0)


frame_count = 0
start_time = time.time()
fps = 0

while(True):
    _, frame = vid.read()
    

    frame_count += 1
    if frame_count % 10 == 0:
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        #print(f"FPS: {fps:.2f}")


    return_frame, is_gaze_facing = gaze_detection.detect(frame)
    
    
    print(f"\nGaze is facing: {is_gaze_facing}")
          
    cv2.imshow('frame',  return_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
     
    if(is_gaze_facing == False):
        pass    # no tracking condition
       
    
    # recall peggiore:
    #code, retrieval_return_string = retrieval.evalFrameTextual(return_frame)

    # recall molto migliore
    
    #code, retrieval_return_string = retrieval.evalFrameTextual(frame)

    # recall massima
    
    retrieval.setUsingMtcnn(True)
    code, retrieval_return_string = retrieval.evalFrameTextual(frame)
    print(f"Retrieval status: {retrieval_return_string}")
    
    if(retrieval.hasMtcnnFailed()):
        print("Retrieval status: Fail or nobody in frame")
        continue
    
    if(retrieval.isPersonBlacklisted()):
        continue    # no tracking condition
    
    print(f"Emotion Detected: {net.predict(return_frame)}\nFPS: {round(fps,2) if frame_count > 10 else '...'}")
    
    
                
    

vid.release()
