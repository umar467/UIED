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

"""
TODO: - Decide how to check components across frames (either by image matching and then position matching, or by getting a binary map and if same sized object appreas in the same image it is probably the same object.)
      - Make the multiple frame detection pipeline.
      - Add Config options for extra component filtering
      - Does it make sense to do a seperate image based component matching if the sift one is already running ? YES
      - Force check if sift points are clustering to see if UI element is being detected their.
"""

class component_detector:
    def __init__(self, config):
        self.config = config
        self.current_frame = []
        self.loaded_frames = 0
        self.data = []

    def get_static_components(self, frame, across_n_frames=10):
        self.get_components(frame)
        if self.loaded_frames < across_n_frames:
            return

        compo_maps= []
        for i in range(across_n_frames):
            old_compo = self.data[-i][1]
            old_compo_map = visualizer.visualize_components(self.current_frame, old_compo, rgb=False, name='s', fill=True, show=False)
            compo_maps.append(old_compo_map)

        common_compos = []

        for component in self.data[-1][1]:
            bbox = component.put_bbox()
            for i in range(across_n_frames):
                if compo_maps[i][bbox[1]:bbox[3], bbox[0]:bbox[2]].mean()<255:
                    break
                common_compos.append(component)

        return common_compos


    def filter_static_components(self, components, static_points):
        if static_points is None or components is None:
            if self.config.logging > 2:
                print("Empty static points array passed.")
            return

        static_components = []
        static_point_image = visualizer.visualize_points(self.current_frame, static_points, rgb=False, show=False, name = 'f')

        for component in components:
           col_min, row_min, col_max, row_max = component.put_bbox()
           mean = static_point_image[row_min:row_max, col_min:col_max].mean()
           if mean > 0:
               static_components.append(component)
        return static_components

    def get_components(self, frame):
        self.current_frame = frame
        frame[3] = det.rm_line(frame[3])
        components = det.component_detection(frame[3], min_obj_area = self.config.min_object_area)
        components = det.merge_intersected_corner(components, frame[1], is_merge_contained_ele=True)
        
        Compo.compos_update(components, frame[1].shape)
        Compo.compos_containment(components)

        #components += rpl.nesting_inspection(frame[1], frame[2], components, ffl_block=self.config.ffl_block)
        #components = det.compo_filter(components, min_area=self.config.min_object_area)
        #Compo.compos_update(components, frame[1].shape)
        
        self.data.append([self.loaded_frames, components])
        self.loaded_frames+=1
        if self.config.logging > 2:
            print(f'{len(components)} components detected in image of size {frame[2].shape}')
        return components