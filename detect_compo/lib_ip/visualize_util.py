#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:02:00 2023

@author: umar
"""

import cv2
import numpy as np
import os

def resize_by_height(org, resize_height):
    w_h_ratio = org.shape[1] / org.shape[0]
    resize_w = resize_height * w_h_ratio
    re = cv2.resize(org, (int(resize_w), int(resize_height)))
    return re

def visualize_points(frame, points, rgb=True, show=True, name="SIFT Point Visualization", scale_down=False):
    if rgb:
        drawing_frame = frame[1].copy()
    else:
        drawing_frame = np.zeros(frame[2].shape)
    if points is None:
        #print("skipping points")
        return drawing_frame
    for point in points:
        cv2.circle(drawing_frame, (int(point[0]),int(point[1])), 5, (255,0,0), thickness=-1, lineType=8, shift=0)
    if show:
        if scale_down:
            drawing_frame = resize_by_height(drawing_frame, 800)
        cv2.imshow(name, drawing_frame)
        cv2.waitKey(10)
    return drawing_frame

def get_json(frame, compos):
    if compos is None:
        return
    if len(compos)==0:
        return

    name = 'Frame '+str(frame[0])
    output = {name: []}
    img_shape = compos[0].image_shape
    output[name].append({'json_format_version':0.1, 'id': 0, 'class': 'Background', 'frequency': 0, 'column_min': 0, 'row_min': 0, 'column_max': img_shape[1],
                             'row_max': img_shape[0], 'width': img_shape[1], 'height': img_shape[0]})
    for compo in compos:
        c = {'id': compo.id, 'class': compo.category}
        c['frequency'] = 8
        (c['column_min'], c['row_min'], c['column_max'], c['row_max']) = compo.put_bbox()
        c['width'] = compo.width
        c['height'] = compo.height
        output[name].append(c)

    return output
def save_json(frame, compos, file_path):
    if compos is None:
        return
    import json
    name = 'Frame '+str(frame[0])
    output = {name: []}
    if os.path.exists(file_path):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    f_out = open(file_path, append_write)

    img_shape = compos[0].image_shape
    output[name].append({'json_format_version':0.1, 'id': 0, 'class': 'Background', 'frequency': 0, 'column_min': 0, 'row_min': 0, 'column_max': img_shape[1],
                             'row_max': img_shape[0], 'width': img_shape[1], 'height': img_shape[0]})
    for compo in compos:
        c = {'id': compo.id, 'class': compo.category}
        c['frequency'] = 8
        (c['column_min'], c['row_min'], c['column_max'], c['row_max']) = compo.put_bbox()
        c['width'] = compo.width
        c['height'] = compo.height
        output[name].append(c)

    json.dump(output, f_out, indent=4)
def visualize_components(frame, components, rgb=True, name='component_visualization', config=None, fill=False, show=True, drawn_frame=None, last_outline = None):
    if rgb:
        drawing_frame = frame[1].copy()
    else:
        drawing_frame = np.zeros(frame[2].shape)
        drawing_frame = drawing_frame.astype(np.uint8)
    if drawn_frame is not None:
        drawing_frame = drawn_frame
    if components==None:
        return
    if fill:
        fill_param = -1
    else:
        fill_param = 2
    for compo in components:
        bbox = compo.put_bbox()
        if config is not None:
            color_map=config.COLOR
            color = color_map[compo.category]
        else:
            color = (255,0,0)
        if not rgb:
            color = (255,255,255)
        drawing_frame = cv2.rectangle(drawing_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, fill_param)
        '''
        if last_outline is not None:
            expected_match = (drawing_frame==drawing_frame).sum()
            got_match = (drawing_frame==last_outline).sum()
            diff = expected_match - got_match
            if diff > 10000:
                cv2.imshow('diff', drawing_frame)
                cv2.waitKey(10)
        '''
    if show:
        cv2.imshow(name, drawing_frame)
        cv2.waitKey(1)
    return drawing_frame
