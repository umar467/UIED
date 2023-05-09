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
import detect_compo.lib_ip.ip_detection as det
from detect_compo.lib_ip.component_detection_utils import component_detector
from detect_compo.lib_ip.ocr_utils import text_extractor
import detect_compo.lib_ip.visualize_util as visualizer
from detect_compo.lib_ip.cnn_utils import cnn
#hello
video_reader_object = video_reader(config)
frame = video_reader_object.get_processed_frame()
static_point_extractor = SIFT_Processor(config)


count = 0
logg = pre.gray_to_gradient(frame[2])
while frame is not None:
    frame = video_reader_object.get_processed_frame()
    #video_reader_object.skip_frames(skip_n=10)


    raw = frame[2]
    #raw = cv2.bilateralFilter(raw, 11, 21, 7)
    raw = pre.gray_to_gradient(raw)
    ogg = raw

    div = 64
    #ogg = ogg // div * div + div // 2
    #logg = logg // div * div + div // 2
    ans = ogg & logg
    #ans = ogg * ans
    #ans = ans.astype(np.uint8)
    #ans = ans*255
    ogg = ans
    #ogg = ogg & logg

    logg = ogg
    count +=1
    if count>8:
        count=0
        logg = raw

        #xn = ogg.mean()
        #ogg = ogg - xn
        #ogg[ogg<80]=0
        #ogg = ogg.astype(np.uint8)

        ogg2 = pre.grad_to_binary(ogg, min=20)

        #ogg2 = cv2.morphologyEx(ogg2, cv2.MORPH_CLOSE, (3,3))  # remove noises
        ogg2 = cv2.dilate(ogg2, None, iterations=2)

        components = det.component_detection(ogg2, min_obj_area=config.min_object_area)
        ogg3 = visualizer.visualize_components(frame, components, show=False, rgb=False)
        visualizer.save_json(frame, components, 'test.json')

        raw2 = pre.grad_to_binary(raw, min=1)
        components = det.component_detection(raw2, min_obj_area=config.min_object_area)
        raw3 = visualizer.visualize_components(frame, components, show=False, rgb=False)

        p5 = np.hstack([ogg, ogg2, ogg3, raw, raw2, raw3])
        #cv2.imshow('f', p5)
        #cv2.imshow('real', buffer[2])
        #cv2.waitKey(100)


