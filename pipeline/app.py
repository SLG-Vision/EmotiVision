from face_retrieval.src.retrieval import Retrieval
from gaze_detection.gaze_detection.main import GazeDetection
from fer_classification.net.FERNet import FERNet
from fer_classification.net.utils import transform
import cv2
import torch

net = FERNet()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net.load_state_dict(torch.load('pipeline/student_distilled.t7', map_location=device))
gaze_detection = GazeDetection(predictor_path="shape_predictor_68_face_landmarks.dat", video=False, print_on_serial=False, crop_frame_paddings=(0.0,0.0,0.0,0.0))
retrieval = Retrieval("def_blacklist.pt", threshold=0.10, debug=True, debugAverage=True, distanceMetric='cosine', usingAverage=False, usingMedian=True, usingMax=False, toVisualize=True, usingMtcnn=False)

fernet_label = {0:'angry', 1:'disgust', 2:'fear', 3:'happy',  4:'sad', 5:'surprise', 6:'neutral'}

retrieval_label = {1:'Detected and identified', 2:'Detected but not identified', 3: 'Not available yet',}

vid = cv2.VideoCapture(0)
while(True):
    _, frame = vid.read()

    return_frame, is_gaze_facing = gaze_detection.detect(frame)
    if(is_gaze_facing == False):
        pass    # no tracking condition
    
    ret = retrieval.evaluateFrame(return_frame)
    if(ret == 3 and retrieval.isUsingMtcnn()):
        continue
    
    if(ret == 1):
        pass    # no tracking condition
    
    input_tensor = transform(return_frame)
    input_tensor = input_tensor.unsqueeze(0)
    pred = net(input_tensor)

    
    print(f"\nGaze is facing:\t{is_gaze_facing}\t\n\tRetrieval status: {retrieval_label[ret]}\t\n\tEmotion Detected: {fernet_label[int(torch.argmax(pred).item())]}")
    cv2.imshow('frame',  return_frame)
                
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()