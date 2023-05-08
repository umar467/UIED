#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 19:54:24 2023

@author: umar
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SIFT_Processor:

    """
    TODO: - Refractor all the point matching code in terms of hashing based array passing. This would enable simultaneous efficient detection of points that are common across frames and points which don't move.
          - See if SIFT points are homogenously located on the screen.
          - If sift point clusters are found, perhaps that is a UI element. Force check each sift point cluster.
    Possible Edge Cases: UI element visible for x frames, where x < frame_buffer.
                         Static detection fails with sudden movement in scene - > sudden drop in sift points being matched across the frame buffer.
    """
    def __init__(self, config):
        self.config = config
        self.data = []  #  [ Frame-1[Keypoints, descriptors] , Frame[keypoints, descriptors]      ]
        self.sift = cv2.SIFT_create()
        self.current_frame = [] # [rgb_frame, grey_frame]
        self.last_frame = []
        self.loaded_frames = 0  # How many frames has this SIFT Processor Object processed so far.
        self.static_objects = [] # X Y coordinates of the detected sift points which are stationary across consecutive frames
        self.statistics = [] # [total_pts_detected, static_pts, dynamic_pts]
        #plt.ion()

    def get_SIFT_Points(self, frame):
        points = []
        featuers, des = self.get_SIFT_features(frame)
        for x in featuers:
            points.append(np.array(x.pt))
        return points
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
        return kp, des
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
    
    def match_points(self, des1, kp1, des2, kp2, static_points):
        
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
                    if des1[m.queryIdx].tobytes() not in static_points:
                        static_points[des1[m.queryIdx].tobytes()] = [np.array(pt1), 1]
                    else:
                        pt , count = static_points[des1[m.queryIdx].tobytes()]
                        if cv2.norm(pt1, pt) < 0.05:
                            count+=1
                            static_points[des1[m.queryIdx].tobytes()] = [np.array(pt1), count]
                    if self.config.log_info:
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
        static_points = {}
        good_points, good_des = self.match_points(self.data[-1][1], self.data[-1][0],self.data[-2][1],self.data[-2][0], static_points)

        for rep in range(2, across_n_frames, 1):
            rep = (rep+1)
            if rep < self.loaded_frames:
                rep = -1*rep
                good_points, good_des = self.match_points(good_points, good_des, self.data[rep][1], self.data[rep][0], static_points)

        static_points = self.process_static_common_points(static_points)
        self.static_objects.append(static_points)
        self.update_stats(good_points, static_points)
        return static_points

    def process_static_common_points(self, static_points):
        static_points = static_points.values()
        static_points = list(static_points)
        counts = []
        points = []
        for pt, count in static_points:
            counts.append(count)
            points.append(pt)
        counts = np.array(counts)
        points = np.array(points)
        indices = np.argwhere(counts == counts.max())
        static_points = points[indices]
        static_points = static_points.squeeze()
        return static_points

    def update_stats(self, good_points, static_points):
        current_frame_total_pts = len(self.data[-1][1])
        total_pts = len(good_points)
        static_pts = len(static_points)
        dynamic_pts = total_pts - static_pts
        self.statistics.append([current_frame_total_pts, total_pts, static_pts, dynamic_pts])
        if self.loaded_frames % 20 == 0:
            p = pd.DataFrame(self.statistics, columns=['current_frame', 'total_common', 'static', 'dynamic'])
            p.plot();
        if self.config.log_info > 2:
            print(f'{len(static_points)} SIFT points with static ratio {static_pts/total_pts}.')
        return