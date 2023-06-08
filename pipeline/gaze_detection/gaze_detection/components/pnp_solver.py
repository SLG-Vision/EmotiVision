import numpy as np
import math
import cv2


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 

def rotationMatrixToEulerAngles(R):
    assert(isRotationMatrix(R))

    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:  # check if it's a gymbal lock situation
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])

    else:  # if in gymbal lock
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


class PnPSolver:
    def __init__(self, calibration=None) -> None:
        self.calibration = calibration
        self.facePOI3d = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
        
    def pose(self, im_size, image_points):  
        
        
        self.axis = np.float32([[200, 0, 0],
                                [0, 200, 0],
                                [0, 0, 200]])
        
        if self.calibration:
            camera_matrix = self.calibration['mtx']  
            dist_coeffs = self.calibration['dist']  
        else:
            # Camera internals
            focal_length = im_size[1]
            center = (im_size[1]/2, im_size[0]/2)
            camera_matrix = np.array(
                                    [[focal_length, 0, center[0]],
                                    [0, focal_length, center[1]],
                                    [0, 0, 1]], dtype = "double"
                                    )
            
            dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

        #print("camera matrix: ")
        #print(camera_matrix)
        #print("distortion coefficients: ")
        #print(dist_coeffs)



        (success, rotation_vector, translation_vector) = cv2.solvePnP(self.facePOI3d, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose
        
        if success:  # if the solvePnP succeed, compute the head pose, otherwise return None

            rotation_vector, translation_vector = cv2.solvePnPRefineVVS(
                self.facePOI3d, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector)
            # this method is used to refine the prediction

            (nose_end_point2D, _) = cv2.projectPoints(
                self.axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            
             # using the Rodrigues formula, this functions computes the Rotation Matrix from the rotation vector
            Rmat = cv2.Rodrigues(rotation_vector)[0]

            pitch, yaw, roll = rotationMatrixToEulerAngles(Rmat) * 180/np.pi


            return nose_end_point2D, pitch, yaw, roll

        else:
            return None, None, None, None
        


