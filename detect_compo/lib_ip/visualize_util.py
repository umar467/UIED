#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:02:00 2023

@author: umar
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
plt.ion()

def resize_by_height(org, resize_height):
    w_h_ratio = org.shape[1] / org.shape[0]
    resize_w = resize_height * w_h_ratio
    re = cv2.resize(org, (int(resize_w), int(resize_height)))
    return re

def visualize_points(frame, points, rgb=True, show=True, name="SIFT Point Visualization", scale_down=False):
    if rgb:
        drawing_frame = frame.copy()
    else:
        drawing_frame = np.zeros(frame.shape)
        drawing_frame = drawing_frame.astype(np.uint8)
    if points is None:
        #print("skipping points")
        return drawing_frame
    for point in points:
        if point.shape == (2,):
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
def visualize_components(frame, components, rgb=True, name='component_visualization', config=None, fill=False, show=True, drawn_frame=None, last_outline = None, offset=0):
    if rgb:
        drawing_frame = frame.copy()
    else:
        drawing_frame = np.zeros(frame.shape)
        drawing_frame = drawing_frame.astype(np.uint8)
    if drawn_frame is not None:
        drawing_frame = drawn_frame
    if components==None:
        return None
    if fill:
        fill_param = -1
    else:
        fill_param = 2
    for compo in components:
        bbox = compo.bbox.put_bbox()
        if config is not None:
            color_map=config.COLOR
            color = color_map[compo.category]
        else:
            color = (255,0,0)
        if not rgb:
            color = (255,255,255)
        drawing_frame = cv2.rectangle(drawing_frame, (bbox[0]+offset, bbox[1]+offset), (bbox[2], bbox[3]), color, fill_param)
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

def visualize_component_crops(frame, components, rgb=True, name='component_visualization', config=None, fill=False, show=True, drawn_frame=None, last_outline = None):
    component_crops = []
    if rgb:
        drawing_frame = frame.copy()
    else:
        drawing_frame = np.zeros(frame.shape)
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
        #drawing_frame = cv2.rectangle(drawing_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, fill_param)
        crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        drawing_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] = crop
        component_crops.append(crop)
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
    component_crops = organize_crop_images(component_crops)
    return drawing_frame, component_crops

def visualize_component_histograms(frame, components):
    crops = get_component_crops_from_frame(frame, components)
    histogram_images = get_histograms_from_comopnent_crops(crops)
    show_crops_and_histograms(crops, histogram_images)
def get_component_crops_from_frame(frame, components):
    component_crops = []
    for compo in components:
        bbox = compo.put_bbox()
        crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        crop = cv2.resize(crop, (128, 128))
        component_crops.append(crop)
    return component_crops


def get_histograms_from_comopnent_crops(component_crops):
    histogram_images = []
    for crop in component_crops:
        histogram_image = None
        for channel in cv2.split(crop):
            histogram_image = histogram_to_image(cv2.calcHist([channel], [0], None, [256], [0, 256]), histogram_image)
        histogram_images.append(histogram_image)
    return histogram_images

# Take opencv histogram object and return a opencv image array
def histogram_to_image(histogram, histImage=None):
    hist_w = 512
    hist_h = 400
    bin_w = int(round( hist_w/256 ))
    if histImage is None:
        histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    cv2.normalize(histogram, histogram, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    for i in range(1, 256):
        cv2.line(histImage, ( bin_w*(i-1), hist_h - int(np.round(histogram[i-1])) ),
                ( bin_w*(i), hist_h - int(np.round(histogram[i])) ),
                ( 255, 255, 255), thickness=2)
    return histImage

# horizontally stack crop image and histogram image and show on screen using oepncv imshow
def show_crops_and_histograms(crops, histogram_images):
    for i in range(len(crops)):
        crop = crops[i]
        histogram_image = histogram_images[i]
        cv2.imshow('crop', crop)
        cv2.imshow('histogram', histogram_image)
        cv2.waitKey(0)

def organize_crop_images(crops):
    max_height = 800
    y = 10
    x = 10
    W = 800
    H = 368
    h = 128
    w = 128
    new_image = np.zeros((W, H))
    for crop in crops:
        # print(crop.shape)
        # print(w)
        # print(h)
        if crop.shape[0]<1:
            break
        if crop.shape[1]<1:
            break
        crop = cv2.resize(crop, (w,h))
        new_image[y:y+h, x:x+w] = crop
        x+=w + 10
        if x + h > H:
            x = 10
            y += w + 10
            if y + w > W:
                break
    new_image = new_image.astype(np.uint8)

    # cv2.imshow('te', new_image)
    # cv2.waitKey(100)
    return new_image


# compare component bbox crops in frame1 and frame2 to see if they have similar SSID score
# Use SSIM to comapre component crops from both frames
# import ssim from skimage.measure
from skimage.metrics import structural_similarity as ssim
def compare_component_crops(frame1, frame2, components):
    component_crops1 = get_component_crops_from_frame(frame1, components)
    component_crops2 = get_component_crops_from_frame(frame2, components)
    scores = []
    for i in range(len(component_crops1)):
        crop1 = component_crops1[i]
        crop2 = component_crops2[i]
        score = ssim(crop1, crop2, multichannel=True)
        print(f'/n/n \n\n\n\n {score} \n')
        if score < 0.8:
            cv2.imshow('cropdiff', np.hstack([crop1, crop2]))
            cv2.waitKey(500)
        scores.append(score)
    return scores

