#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:02:00 2023

@author: umar
"""

import cv2
import numpy as np

def visualize_points(frame, points, rgb=True):
    if rgb:
        drawing_frame = frame[1].copy()
    else:
        drawing_frame = np.zeros(frame[2].shape)
    if points is None:
        #print("skipping points")
        return drawing_frame
    for point in points:
        cv2.circle(drawing_frame, (int(point[0]),int(point[1])), 5, (255,0,0), thickness=-1, lineType=8, shift=0)
    cv2.imshow("SIFT Point Visualization", drawing_frame)
    cv2.waitKey(10)
    return drawing_frame
    
def visualize_components(frame, components, rgb=True, name='component_visualization', config=None):
    if rgb:
        drawing_frame = frame[1].copy()
    else:
        drawing_frame = np.zeros(frame[2].shape)
    for compo in components:
        bbox = compo.put_bbox()
        if config is not None:
            color_map=config.COLOR
            color = color_map[compo.category]
        else:
            color = (255,0,0)
        drawing_frame = cv2.rectangle(drawing_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    cv2.imshow(name, drawing_frame)
    cv2.waitKey(10)
    return drawing_frame