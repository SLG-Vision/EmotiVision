import cv2, os

class HaarCascade():
    def __init__(self) -> None:
        self.face_cascade = cv2.CascadeClassifier(os.path.join('predictors','haarcascade_frontalface_alt.xml'))
        self.eye_cascade = cv2.CascadeClassifier(os.path.join('predictors','haarcascade_eye.xml'))

    def face_detection(self, frame, scale_factor=1.1, min_neighbors=2):
        """
        https://stackoverflow.com/questions/15403850/opencv-how-to-improve-accuracy-of-eyes-detection-using-haar-classifier-cascade
        The scaleFactor parameter is used to determine how many different sizes of eyes the function will look for. Usually this value is 1.1 for the best detection. Setting this parameter to 1.2 or 1.3 will detect eyes faster but doesn't find them as often, meaning the accuracy goes down.
        minNeighbors is used for telling the detector how sure he should be when detected an eye. Normally this value is set to 3 but if you want more reliability you can set this higher. Higher values means less accuracy but more reliability
        The flags are used for setting specific preferences, like looking for the largest object or skipping regions. Default this value = 0. Setting this value can make the detection go faster
        """            
        return self.face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scale_factor, min_neighbors)
    
    def face_framing(self, faces, frame):
        for (x, y, w, h) in faces:            
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        return frame  
    
    
    def eye_detection(self, faces, frame):
        eyes_ret = []
        for (x, y, w, h) in faces:
            roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)

            eyes_ret.append(eyes)
                
        return eyes_ret
    
    def eye_framing(self, faces, frame):
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
                
        return frame