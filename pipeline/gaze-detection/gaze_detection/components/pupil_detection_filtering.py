import cv2
import numpy as np
import random

BLUR_RADIUS = 5
CONTRAST_VALUE = 1.25
EROSION_SIZE = 3

class PupilDetection():
    def __init__(self, visual_verbose: bool = False) -> None:
        """Classical approach to pupil detection using simple optical transformations. 
        The image is trasformed using several filters and normalized, to highlight the pupil darkness.
        Finally, in the binarized image the pupil is found searching for the biggest contours.

        Args:
            visual_verbose (bool, optional): condition according to which the steps images are showed. Defaults to False. Defaults to False.
        """
        self.verbose = visual_verbose
    
    def _show_if(self, title: str = "Frame", frame: np.ndarray = None):
        
        """Function that show an image only if needed.

        Args:
            title (str, optional): title of the Frame. Defaults to "Frame".
            frame (np.ndarray, optional): the actual image. Defaults to None.
        """
        
        if (self.verbose and frame):
            cv2.imshow(title, frame)
        else:
            pass
    
    def detect_pupil(self, eye: np.ndarray):
        """Classical approach to pupil detection using simple optical transformations. 
        The image is trasformed using several filters and normalized, to highlight the pupil darkness.
        Finally, in the binarized image the pupil is found searching for the biggest contours.

        Args:
            eye (np.ndarray): The actual image (the condition where the method works well is that the given image is just of the eye without paddings).
            verbose (bool, optional): The verbose flag, that set if step images are shown during the process. Defaults to False.

        Returns:
            (x, y): positions of the pupil in the image (relative to the box of the given image)
        """

        if len(eye.shape)>2:
            eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        
        eye = np.clip(eye * CONTRAST_VALUE, 0, 255)
        eye = eye.astype('uint8')
        eye = cv2.equalizeHist(eye)
        
        self._show_if("Equalized and contrasted eye", eye)

        blurred_eye = cv2.GaussianBlur(eye, (BLUR_RADIUS, BLUR_RADIUS), 0)
        
        kernel = np.ones((EROSION_SIZE,EROSION_SIZE),np.uint8)
        erosion = cv2.erode(blurred_eye,kernel,iterations = 1)

        self._show_if("Blurred and eroded eye", erosion)

        threshold = cv2.adaptiveThreshold(erosion,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,51,-5)
        
        self._show_if("Thresholded eye", threshold)
        
        try:
            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
        except:    
            print('WARNING: Error during moments and contours detection.')
            
            centroid_x = None
            centroid_y = None
        
        return (centroid_x, centroid_y)