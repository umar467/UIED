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
sift_processor = SIFT_Processor(config)
#text_extractor = text_extractor(config)
#classifier = cnn(config)

while(frame is not None):
    
    frame = video.get_processed_frame()
    sps = sift_processor.get_static_objects(frame)
    #sift_processor.get_homography()
    
    #video.skip_frames(10)
    #detected_compos = component_detector.get_components(frame)
    
    #classifier.process(frame, detected_compos)
    
    visualizer.visualize_points(frame, sps)
    
    #visualizer.visualize_components(frame, detected_compos, config=config)
    