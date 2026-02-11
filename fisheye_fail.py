import cv2
import numpy as np
import json

def main():
    with open('data/fisheye-failure-case.json', 'r') as file:
        data = json.load(file)
        object_points = [np.asarray(pts) for pts in data['object points']]
        image_points = [np.asarray(pts) for pts in data['image points']]
        image_size = tuple(data['image size'])
        flags = data['calibration flags']

    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(object_points, image_points, image_size, None, None, flags=flags)
    print(rms)
    print(K)
    print(D)


if __name__ == '__main__':
    main()
