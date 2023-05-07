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

video  = video_reader(config)
frame = video.get_processed_frame()
component_detector = component_detector(config)
static_point_extractor = SIFT_Processor(config)
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