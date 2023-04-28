#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 19:54:24 2023

@author: umar
"""

import cv2
import numpy as np

class SIFT_Processor:

    """
    TODO: - Refractor all the point matching code in terms of hashing based array passing. This would enable simultaneous efficient detection of points that are common across frames and points which don't move.
    """
    def __init__(self, config):
        self.config = config
        self.data = []  #  [ Frame-1[Keypoints, descriptors] , Frame[keypoints, descriptors]      ]
        self.sift = cv2.SIFT_create()
        self.current_frame = [] # [rgb_frame, grey_frame]
        self.last_frame = []
        self.loaded_frames = 0  # How many frames has this SIFT Processor Object processed so far.
        self.static_objects = [] # X Y coordinates of the detected sift points which are stationary across consecutive frames
        
    def get_SIFT_features(self, frame):
        '''
        Input: greyscale frame
        Output: Append the detected SIFT keypoints and feature descriptors to the objects data array
        '''
        self.last_frame = self.current_frame
        self.current_frame = frame
        kp, des = self.sift.detectAndCompute(frame[1], None)
        self.data.append([kp, des])
        self.loaded_frames+=1

    # Not Used but Functional
    def get_homography(self):
        if self.loaded_frames == 1:
            self.last_frame = self.current_frame
            return
        import numpy as np
        import cv2
        from matplotlib import pyplot as plt
        
        MIN_MATCH_COUNT = 10
        
        #img1 = cv2.imread('box.png',0)          # queryImage
        #img2 = cv2.imread('box_in_scene.png',0) # trainImage
        
        img1 = self.current_frame[1]
        img2 = self.last_frame[1]
        
        # Initiate SIFT detector
        #sift = cv2.SIFT()
        
        # find the keypoints and descriptors with SIFT
        #kp1, des1 = sift.detectAndCompute(img1,None)
        #kp2, des2 = sift.detectAndCompute(img2,None)
        
        kp1, des1 = self.data[-1]
        kp2, des2 = self.data[-2]
        
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des1,des2,k=2)
        
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
                
        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            mask = 1-mask
            matchesMask = mask.ravel().tolist()
        
            h,w,_ = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            print(dst)
            print(M)
            img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        
        else:
            print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
            matchesMask = None
            
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
        
        #plt.imshow(img2, 'gray'),plt.show()
        cv2.imshow('hg', img3)
        cv2.waitKey(30)
    
    def match_points(self, des1, kp1, des2, kp2, match_points, match_point_locations):
        
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good_points = []
        good_des = []
        """
        Good Points is all common points within the .7 sift match rule
        Match Points is all the points that are common and don't move 
        """

        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                good_points.append(des1[m.queryIdx])
                good_des.append(kp1[m.queryIdx])
                pt1 = kp1[m.queryIdx].pt
                pt2 = kp2[m.trainIdx].pt
                dis = cv2.norm(pt1,pt2)
                if dis<0.05:
                    match_points.append(des1[m.queryIdx])
                    match_point_locations.append(pt1)
                    if self.config.logging > 4:
                        print(i, pt1,pt2, dis)
        
        return np.array(good_points), good_des
    
    def get_static_objects(self, frame, across_n_frames=10):
        """
        Input: greyscale Frame
        Output: Appends the sift keypoints stationary across frames to the array static objects
        """
        self.get_SIFT_features(frame)
        
        if self.loaded_frames == 1:
            self.static_objects.append([])
            return
        match_points = []
        match_point_locations = []
        good_points, good_des = self.match_points(self.data[-1][1], self.data[-1][0],self.data[-2][1],self.data[-2][0], match_points, match_point_locations)

        for rep in range(2, across_n_frames, 1):
            rep = (rep+1)
            if rep < self.loaded_frames:
                rep = -1*rep
                good_points, good_des = self.match_points(good_points, good_des, self.data[rep][1], self.data[rep][0], match_points, match_point_locations)

        match_points = np.array(match_points)
        unique, indices, counts = np.unique(match_points, return_counts=True, axis=0, return_index=True)

        max_indices = np.argwhere(counts==counts.max())

        static_points = np.array(match_point_locations)[max_indices]
        static_points=static_points.reshape(-1, 2)
        static_to_dynamic_point_ratio = counts.min() / counts.max()
        if static_to_dynamic_point_ratio > 0.8:
            static_points = []
        print(f'{len(static_points)} static SIFT points discovered from last frame with static to moving ratio of {counts.min() / counts.max()}')
        self.static_objects.append(static_points)
        
        if self.config.logging > 2:
            print(f'{len(static_points)} static SIFT points discovered from last frame with static to moving ratio of {counts.min()/counts.max()}')
            
        return static_points