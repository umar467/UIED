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


def new_ui_save(frame0, framel, config):

    if not os.path.exists(config.output_folder + '/uis/'):
        os.mkdir(config.output_folder + '/uis/')
    cv2.imwrite(config.output_folder + '/uis/' + str(config.current_ui_number) + '.png', frame0)
    config.current_ui_number += 1
    cv2.imwrite(config.output_folder + '/uis/' + str(config.current_ui_number) + '.png', framel)
    config.current_ui_number += 1
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
        cv2.waitKey(500)
    return drawing_frame

def visualize_components_accumulative(frame, components, text = False):
    frame = np.zeros(frame.shape)
    frame = frame.astype(np.float64)
    for compo in components:
        if text:
            if compo.category == 'Text':
                bbox = compo.bbox.put_bbox()
                for xx in compo.detected_in_frames:
                    frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] += 255
        else:
            if compo.category != 'Text':
                bbox = compo.bbox.put_bbox()
                for xx in compo.detected_in_frames:
                    frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] += 255
    return frame


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
def show_crops_and_histograms(crops, histogram_images, config):
    shape = (128,128)
    crop = crops[-1]
    crop = cv2.resize(crop, shape)
    histogram = histogram_images[-1]
    histogram = cv2.resize(histogram, shape)
    big_image = np.hstack([crop, histogram])
    no = 5
    if len(crops) < no:
        no = len(crops)
    for i in range(no):
        crop = crops[i]
        crop = cv2.resize(crop, shape)
        histogram = histogram_images[i]
        histogram = cv2.resize(histogram, shape)
        current = np.hstack([crop, histogram])
        big_image = np.vstack([big_image, current])

    cv2.imwrite(config.output_folder + 'crops_hists.png', big_image)
    # cv2.imshow('crops and histograms', big_image)
    # cv2.waitKey(0)

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


def assign_red_to_top_percentile(image):
    # Calculate the threshold value for the top 20% percentile
    percentile_threshold = np.percentile(image, 95)

    # Convert grayscale image to color image
    color_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Assign red color (0, 0, 255) to the pixels above the threshold
    color_image[np.where(image > percentile_threshold)] = [0, 0, 255]
    
    # assign green , red and blue color to pixels in three equal treshold ranges
    color_image[np.where(image > percentile_threshold/3)] = [0, 255, 0]
    color_image[np.where(image > percentile_threshold/3*2)] = [255, 0, 0]
    color_image[np.where(image > percentile_threshold)] = [0, 0, 255]
    color_image[np.where(image < percentile_threshold/3)] = [0, 0, 0]
    return color_image





def Save_plots_and_heatmpas(JSON_Processor, compos, frame, config):
    frame = visualize_components_accumulative(frame, compos, text=True)
    mean_image = cv2.GaussianBlur(frame, (25, 25), 0)
    #cv2.imwrite(config.output_folder + 'text_heatmap.png', assign_red_to_top_percentile(mean_image))

    frame = visualize_components_accumulative(frame, compos)
    mean_image = cv2.GaussianBlur(frame, (25, 25), 0)
    #cv2.imwrite(config.output_folder + 'non_text_heatmap.png', assign_red_to_top_percentile(mean_image))

    import pandas as pd
    fd = JSON_Processor.get_stats()
    p = pd.DataFrame(fd[0], columns=['total_detected', 'area_filtered', 'overlap_filtered', 'sift_filtered'])
    plot = p.plot(title='compo detection stats')
    plot.set_xlabel("Frames x 10")
    plot.set_ylabel("Frequency")
    fig = plot.get_figure()
    fig.savefig(config.output_folder + "component_stats.png")
    plt.close()


    p = pd.DataFrame(fd[1], columns=['total_detected', 'filtered'])
    plot = p.plot(title='database filter stats');
    plot.set_xlabel("Frames x 10")
    plot.set_ylabel("Frequency")
    fig = plot.get_figure()
    fig.savefig(config.output_folder + "database_stats.png")
    plt.close()
