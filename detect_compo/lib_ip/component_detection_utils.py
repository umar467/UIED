#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:51:24 2023

@author: umar
"""

import numpy as np
import cv2
import detect_compo.lib_ip.ip_detection as det
import detect_compo.lib_ip.Component as Compo
import detect_compo.ip_region_proposal as rpl


class component_detector:
    
    def __init__(self, config):
        self.config = config
        self.current_frame = []
        self.loaded_frames = 0
        self.data = []
        
    def get_components(self, frame):
        self.current_frame = frame
        frame [3] = det.rm_line(frame[3])
        components = det.component_detection(frame[3], min_obj_area = self.config.min_object_area)        
        components = det.merge_intersected_corner(components, frame[1], is_merge_contained_ele=True)
        
        Compo.compos_update(components, frame[1].shape)
        Compo.compos_containment(components)
        
        components += rpl.nesting_inspection(frame[1], frame[2], components, ffl_block=self.config.ffl_block)
        components = det.compo_filter(components, min_area=self.config.min_object_area)
        Compo.compos_update(components, frame[1].shape)
        
        
        self.data.append([self.loaded_frames, components])
        self.loaded_frames+=1
        if self.config.logging > 2:
            print(f'{len(components)} components detected in image of size {frame[2].shape}')
        return components