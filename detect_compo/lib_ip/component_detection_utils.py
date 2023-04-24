#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:51:24 2023

@author: umar
"""

import numpy as np
import cv2
import detect_compo.lib_ip.ip_detection as det


class component_detector:
    
    def __init__(self, config):
        self.config = config
        self.current_frame = []
        self.loaded_frames = 0
        self.data = []
        
    def get_components(self, frame):
        self.current_frame = frame
        components = det.component_detection(frame[3], min_obj_area = self.config.min_object_area)        
        self.data.append([self.loaded_frames, components])
        self.loaded_frames+=1
        if self.config.logging > 2:
            print(f'{len(components)} components detected in image of size {frame[2].shape}')
        return components