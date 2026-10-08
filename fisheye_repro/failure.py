import numpy as np
import cv2
import yaml
import itertools
import functools

#cv2.fisheye.calibrate(


with open('broken.yaml', 'r') as file:
    data = yaml.safe_load(file)

#for d in data:


all_corners = []
all_obj_points = []

detections = data['detections']
image_size = data['image_size']

def combine(first, second):
    d = dict()
    for (k1, v1), (k2, v2) in zip(first.items(), second.items()):
        d[k1] = v2
    return d

d = functools.reduce(combine, detections)

if True:
    #print(d)
    corners = d['corners']
    ids = d['ids']
    object_points = np.pad(np.asarray(d['object_points']), ((0, 0), (0, 1)))#[..., np.newaxis, :]

    for i, pts in zip(ids,  map(np.asarray, corners)):
        #print(pts.shape)
        all_corners += [pts]
        all_obj_points += [object_points[i]]


print(all_obj_points)
#all_obj_points = np.copy(np.pad(np.vstack(all_obj_points), ((0, 0), (0, 1))).reshape((-1, 1, 3)))
#all_corners = np.vstack(all_corners).reshape(-1, 2)

cv2.fisheye.calibrate(all_obj_points, all_corners, image_size, None, None)
