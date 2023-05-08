#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 18:17:24 2023

@author: umar
"""

import cv2
import numpy as np
import detect_compo.lib_ip.ip_draw as draw
import detect_compo.lib_ip.ip_preprocessing as pre
from detect_compo.lib_ip.Component import Component
import detect_compo.lib_ip.Component as Compo
from config.NEW_CONFIG_UIED import Configuration
config = Configuration()
from detect_compo.lib_ip.video_utils import video_reader
from detect_compo.lib_ip.SIFT_utils import SIFT_Processor
from detect_compo.lib_ip.component_detection_utils import component_detector
from detect_compo.lib_ip.ocr_utils import text_extractor
import detect_compo.lib_ip.visualize_util as visualizer
from detect_compo.lib_ip.cnn_utils import cnn
#hello
video_reader_object = video_rdfdeader(config)
frame = video_reader_object.get_processed_frame()
static_point_extractor = SIFT_Processor(config)

'''
while frame is not None:
    frame = video_reader_object.get_processed_frame()
    show = frame[2].astype(np.float32)
    for i in range(50):
        frame = video_reader_object.get_processed_frame()
        show = show+frame[2]
    show[show<show.mean()]=0
    show *= (255.0 / show.max())
    pres = show.copy()
    #pres[pres<pres.mean()] = 0

    #pres *= (255.0 / pres.max())

    cv2.imshow('diff', pres)
    #cv2.imshow('real', buffer[2])
    cv2.waitKey(10)
'''
'''
show = frame[1]
while frame is not None:
    frame = video_reader_object.get_processed_frame()
    show = show&frame[1]
    if show.mean() < 20:
        show = frame[1]
        print('new lease on life')
    cv2.imshow('diff', show)
    #cv2.imshow('real', buffer[2])
    cv2.waitKey(10)

'''
'''
buffer = [video_reader_object.get_processed_frame() for x in range(15)]
buffer = [x[2] for x in buffer]
while frame is not None:
    show = buffer[0]&buffer[1]&buffer[2]&buffer[3]&buffer[4]&  buffer[5]&buffer[6]&buffer[7]&buffer[8]&buffer[9]
    buffer = [video_reader_object.get_processed_frame() for x in range(15)]
    buffer = [x[2] for x in buffer]

    #pres = show.astype(np.uint8)
    #pres = pres*255
    #pres = cv2.resize(pres, (800,368))
    cv2.imshow('diff', show)
    cv2.imshow('real', buffer[2])
    cv2.waitKey(10)
'''
#allpoints = static_point_extractor.get_SIFT_Points(frame)
#static_points = static_point_extractor.get_static_objects(frame, across_n_frames=5)
#visualizer.visualize_points(frame, static_points, show=True, scale_down=True)

''''
component_detector = component_detector(config)
#text_extractor = text_extractor(config)
classifier = cnn(config)
video.skip_frames(30)
outline = None
while frame is not None:
    
    frame = video.get_processed_frame()
    static_points = static_point_extractor.get_static_objects(frame, across_n_frames=5)
    #sift_processor.get_homography()

    detected_compos = component_detector.get_static_components(frame, across_n_frames=1)
    #detected_compos = component_detector.get_components(frame)
    detected_compos = component_detector.filter_static_components(detected_compos, static_points)

    if detected_compos is not None:
        classifier.process(frame, detected_compos)
    #video.skip_frames(5)
    drawn_frame = visualizer.visualize_points(frame, static_points, show=False)
    
    visualizer.visualize_components(frame, detected_compos, config=config, drawn_frame = drawn_frame)
    if outline is None:
        outline = visualizer.visualize_components(frame, detected_compos, config=config,rgb=False, name = 'outline',fill=True)
    else:
        outline = visualizer.visualize_components(frame, detected_compos, config=config, rgb=False, name='outline', last_outline = outline,
                                                  fill=True)
    visualizer.save_json(frame, detected_compos, 'sample.json')

'''