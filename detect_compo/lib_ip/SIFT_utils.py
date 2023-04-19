#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 19:54:24 2023

@author: umar
"""

import cv2
import numpy as np

class SIFT_Processor:
    
    def __init__(self, config):
        self.config = config
        self.data = []  #  [ Frame[Keypoints, descriptors] , Frame+1[keypoints, descriptors]      ]
        self.sift = cv2.SIFT_create()
        self.current_frame = [] # [rgb_frame, grey_frame]
        self.loaded_frames = 0  # How many frames has this SIFT Processor Object processed so far.
        self.static_objects = [] # X Y coordinates of the detected sift points which are stationary across consecutive frames
        
    def get_SIFT_features(self, frame):
        '''
        Input: greyscale frame
        Output: Append the detected SIFT keypoints and feature descriptors to the objects data array
        '''
        self.current_frame = frame
        kp, des = self.sift.detectAndCompute(frame[1], None)
        self.data.append([kp, des])
        self.loaded_frames+=1
        
    def get_static_objects(self, frame):
        '''
        Input: greyscale Frame
        Output: Appends the sift keypoints stationary across frames to the array static objects
        '''
        self.get_SIFT_features(frame)
        
        if self.loaded_frames == 1:
            self.static_objects.append([])
            return
        
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(self.data[-1][1],self.data[-2][1],k=2)
        match_points = []
        
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                
                pt1 = self.data[-1][0][m.queryIdx].pt
                pt2 = self.data[-2][0][m.trainIdx].pt
                dis = cv2.norm(pt1,pt2)
                if dis<0.05:
                    match_points.append(pt1)
                    if self.config.logging > 4:
                        print(i, pt1,pt2, dis)
        self.data.append(match_points)
        
        if self.config.logging > 2:
            print(f'{len(match_points)} static SIFT points discovered from last frame.')
            
        return match_points