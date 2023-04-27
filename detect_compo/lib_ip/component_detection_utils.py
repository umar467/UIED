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
import detect_compo.lib_ip.visualize_util as visualizer


class component_detector:
    
    def __init__(self, config):
        self.config = config
        self.current_frame = []
        self.loaded_frames = 0
        self.data = []
    
    def make_sift_image(self, frame, sps):
        sift_image = visualizer.visualize_points(frame, sps, rgb=False)
        return sift_image
        
    
    def get_components(self, frame, sps):
        self.current_frame = frame
        frame [3] = det.rm_line(frame[3])
        components = det.component_detection(frame[3], min_obj_area = self.config.min_object_area)        
        components = det.merge_intersected_corner(components, frame[1], is_merge_contained_ele=True)
        
        Compo.compos_update(components, frame[1].shape)
        Compo.compos_containment(components)
        
        #components += rpl.nesting_inspection(frame[1], frame[2], components, ffl_block=self.config.ffl_block)
        #components = det.compo_filter(components, min_area=self.config.min_object_area)
        #Compo.compos_update(components, frame[1].shape)
        
        sift_filtered_components = []
        sift_image = self.make_sift_image(frame, sps)
        drawing_frame = np.zeros(frame[1].shape)


        if sps is not None:
            for component in components:
                col_min, row_min, col_max, row_max = component.put_bbox()
                width = col_max - col_min
                height = row_max - row_min
                mean = sift_image[row_min:row_max, col_min:col_max].mean()
                #print(mean)
                if mean > 0:
                    sift_filtered_components.append(component)
                    drawing_frame[row_min:row_max, col_min:col_max] = frame[1][row_min:row_max, col_min:col_max]
        components = sift_filtered_components
        
        cv2.imshow('cropped view', drawing_frame)
        cv2.waitKey(10)
        
        self.data.append([self.loaded_frames, components])
        self.loaded_frames+=1
        if self.config.logging > 2:
            print(f'{len(components)} components detected in image of size {frame[2].shape}')
        return components