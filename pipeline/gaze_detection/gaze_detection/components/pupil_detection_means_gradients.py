import numpy as np
import cv2
from math import sqrt
from queue import *

GRADIENT_THRESHOLD_VALUE = 10
BLUR_SIZE = 5
POSTPROCESSING_THRESHOLD_VALUE = 0.9
MAX_EYE_SIZE = 10


class PupilDetection():
    def __init__(self, postprocessing=False) -> None:
        self.postprocessing = postprocessing
        pass

    def _test_possible_centers(self, x, y, gx, gy, out, weight):
        rows, cols = out.shape
        for row in range(rows):
            for col in range(cols):
                dx, dy = x-col, y-row

                if dx == 0 and dy == 0:
                    continue

                magnitude = sqrt((dx*dx)+(dy*dy))
                dx = dx/magnitude
                dy = dy/magnitude

                dot_product = max(0.0, dx * gx + dy * gy)

                out[row][col] += dot_product * dot_product * weight[row][col]

    def _get_magnitude_matrix(self, mat_x, mat_y):
        rows, cols, = np.shape(mat_x)
        matrix = np.zeros((rows, cols), dtype=np.float32)
        for row in range(rows):
            for col in range(cols):
                gx = mat_x[row][col]
                gy = mat_y[row][col]
                matrix[row][col] = sqrt((gx*gx)+(gy*gy))
        return matrix

    def _compute_dynamic_threshold(self, magnitude_matrix):
        (meanMagnGrad, meanMagnGrad) = cv2.meanStdDev(magnitude_matrix)
        stdDev = meanMagnGrad[0] / \
            sqrt(magnitude_matrix.shape[0]*magnitude_matrix.shape[1])
        return GRADIENT_THRESHOLD_VALUE*stdDev+meanMagnGrad[0]

    def _flood_should_push_point(self, dir, mat):
        px, py = dir
        rows, cols = np.shape(mat)
        if px >= 0 and px < cols and py >= 0 and py < rows:
            return True
        else:
            return False

    def _flood_kill_edges(self, mat):
        rows, cols = np.shape(mat)
        cv2.rectangle(mat, (0, 0), (cols, rows), 255)
        mask = np.ones((rows, cols), dtype=np.uint8)
        mask = mask * 255
        to_do = Queue()
        to_do.put((0, 0))
        while to_do.qsize() > 0:
            px, py = to_do.get()
            if mat[py][px] == 0:
                continue
            right = (px + 1, py)
            if self._flood_should_push_point(right, mat):
                to_do.put(right)
            left = (px - 1, py)
            if self._flood_should_push_point(left, mat):
                to_do.put(left)
            down = (px, py + 1)
            if self._flood_should_push_point(down, mat):
                to_do.put(down)
            top = (px, py - 1)
            if self._flood_should_push_point(top, mat):
                to_do.put(top)
            mat[py][px] = 0.0
            mask[py][px] = 0
        return mask

    def _compute_gradient(self, img):
        out = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        if img.shape[0] < 2 or img.shape[1] < 2:
            print("EYES too small")
            return out
        for y in range(0, out.shape[0]):
            out[y][0] = img[y][1]-img[y][0]
            for x in range(1, out.shape[1]-1):
                out[y][x] = (img[y][x+1]-img[y][x-1])/2.0
            out[y][out.shape[1]-1] = img[y][out.shape[1]-1] - \
                img[y][out.shape[1]-2]
        return out

    def detect_pupil(self, eye_image):
        if (len(eye_image.shape) <= 0 or eye_image.shape[0] <= 0 or eye_image.shape[1] <= 0):
            return (0, 0)
        eye_image = eye_image.astype(np.float32)
        scale_value = 1.0
        if (eye_image.shape[0] > MAX_EYE_SIZE or eye_image.shape[1] > MAX_EYE_SIZE):
            scale_value = max(
                MAX_EYE_SIZE/float(eye_image.shape[0]), MAX_EYE_SIZE/float(eye_image.shape[1]))
            eye_image = cv2.resize(
                eye_image, None, fx=scale_value, fy=scale_value, interpolation=cv2.INTER_AREA)

        # compute gradient for each point
        grad_arr_x = self._compute_gradient(eye_image)
        grad_arr_y = np.transpose(
            self._compute_gradient(np.transpose(eye_image)))

        magnitude_matrix = self._get_magnitude_matrix(grad_arr_x, grad_arr_y)
        # find a threshold, element below that will be put to 0, scaled otherwise
        gradient_threshold = self._compute_dynamic_threshold(magnitude_matrix)

        for y in range(eye_image.shape[0]):
            for x in range(eye_image.shape[1]):
                if (magnitude_matrix[y][x] > gradient_threshold):
                    grad_arr_x[y][x] = grad_arr_x[y][x]/magnitude_matrix[y][x]
                    grad_arr_y[y][x] = grad_arr_y[y][x]/magnitude_matrix[y][x]
                else:
                    grad_arr_x[y][x] = 0.0
                    grad_arr_y[y][x] = 0.0

        # create weights
        weight = cv2.GaussianBlur(eye_image, (BLUR_SIZE, BLUR_SIZE), 0)

        # invert the weight matrix
        for y in range(weight.shape[0]):
            for x in range(weight.shape[1]):
                weight[y][x] = 255-weight[y][x]

        out_sum = np.zeros(
            (eye_image.shape[0], eye_image.shape[1]), dtype=np.float32)
        out_sum_rows, out_sum_cols = np.shape(out_sum)

        # test every possible center
        for row in range(out_sum_rows):
            for col in range(out_sum_cols):
                gx = grad_arr_x[row][col]
                gy = grad_arr_y[row][col]
                if gx == 0.0 and gy == 0:
                    continue
                self._test_possible_centers(col, row, gx, gy, out_sum, weight)

        num_gradients = weight.shape[0]*weight.shape[1]
        out = np.divide(out_sum, num_gradients*10)

        _, max_val, _, max_p = cv2.minMaxLoc(out)

        # post_processing
        if self.postprocessing:
            flood_threshold = max_val*POSTPROCESSING_THRESHOLD_VALUE
            ret, flood_clone = cv2.threshold(
                out, flood_threshold, 0.0, cv2.THRESH_TOZERO)
            mask = self._flood_kill_edges(flood_clone)
            _, max_val, _, max_p = cv2.minMaxLoc(out, mask)

        max_p = (int(max_p[0]/scale_value), int(max_p[1]/scale_value))
        return max_p
