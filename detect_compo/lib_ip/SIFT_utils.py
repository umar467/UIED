#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 19:54:24 2023

@author: umar
"""

import mkl
# mkl.set_num_threads(1)
import cv2
# cv2.setNumThreads(1)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from config.CONFIG import Configuration
# config = Configuration()
import plotly.express as px

class SIFT_Processor:

    def __init__(self, config):
        self.config = config
        self.data = []  #  [ Frame-1[Keypoints, descriptors] , Frame[keypoints, descriptors]      ]
        self.sift = cv2.SIFT_create()
        self.current_frame = [] # [rgb_frame, grey_frame]
        self.last_frame = []
        self.loaded_frames = 0  # How many frames has this SIFT Processor Object processed so far.
        self.static_objects = [] # X Y coordinates of the detected sift points which are stationary across consecutive frames
        self.statistics = [] # [total_pts_detected, static_pts, dynamic_pts]
        plt.ion()
        self.global_distance = 0
        self.current_ui_number = 0


    def get_SIFT_features(self, frame):
        '''
        Input: greyscale frame
        Output: Append the detected SIFT keypoints and feature descriptors to the objects data array
        '''
        self.last_frame = self.current_frame
        self.current_frame = frame
        kp, des = self.sift.detectAndCompute(frame, None)
        if des.shape[0] > self.config.maximum_SIFT_points_per_frame:
            if self.config.log_warnings:
                print(f'High SIFT featuers {len(kp)} Detected !!')
            kp = (kp[0], kp[1])
            des = des[0:2]
        self.data.append([kp, des])
        self.loaded_frames+=1
        #self.show_SIFT_points(frame, des, kp, name='SIFTs', use_cv=True)
        return kp, des
    # Not Used but Functional
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

        distance_average = 0
        distance_average_count = 0
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                good_points.append(des1[m.queryIdx])
                good_des.append(kp1[m.queryIdx])
                pt1 = kp1[m.queryIdx].pt
                pt2 = kp2[m.trainIdx].pt
                dis = cv2.norm(pt1,pt2)
                if dis<0.05:
                    if des1[m.queryIdx].tobytes() not in static_points:
                        static_points[des1[m.queryIdx].tobytes()] = [np.array(pt1), 1, kp1[m.queryIdx].size]
                    else:
                        pt , count, size = static_points[des1[m.queryIdx].tobytes()]
                        if cv2.norm(pt1, pt) < 0.05:
                            count+=1
                            static_points[des1[m.queryIdx].tobytes()] = [np.array(pt1), count, size]
                    if self.config.log_info:
                        print(i, pt1,pt2, dis)
                else:
                    distance_average+=dis
                    distance_average_count+=1
        if distance_average_count>0:
            self.global_distance += distance_average/distance_average_count
            #print(self.global_distance)
        
        return np.array(good_points), good_des

    def plot_SIFT_detection_plots(self):
        p = pd.DataFrame(self.statistics, columns=['current_frame', 'total_common', 'static', 'dynamic'])
        plot = p.plot();
        #plot.title('SIFT Features across Frames')
        plot.set_xlabel("Frames x 10")
        plot.set_ylabel("Frequency")
        fig = plot.get_figure()
        # video_name = 'video_' + str(self.config.input_video)
        # video_name = video_name.replace('/', '_')
        # video_name = video_name.replace('.mp4', '/')
        import os
        if not os.path.exists(self.config.output_folder):
            os.mkdir(self.config.output_folder)
        output_path = self.config.output_folder + 'sift.png'
        fig.savefig(output_path)
        plt.close()

    def get_SIFT_point_mask(self, static_points):
        # draw_frame = cv2.drawKeypoints(frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # compare results and implementation
        rframe = self.current_frame
        frame = np.zeros_like(rframe)
        for pt, size in static_points:
            pt = tuple(pt.astype(int))
            if size<5:size=size*10
            if size<10:size=size*5
            if size<15:size=size*2
            frame = cv2.circle(frame, pt, int(size), (255, 255, 255), cv2.FILLED, 8,0)
        return frame

    def get_static_pixels(self, frames, JSON_Processor):
        for frame in frames:
            self.get_SIFT_features(frame)
        static_pixels, new_ui = self.get_static_objects(across_n_frames=self.config.frame_buffer_size)
        JSON_Processor.add_sift_statistics_to_current_frame(self.statistics)
        self.plot_SIFT_detection_plots()
        distance_average = self.global_distance/self.loaded_frames
        JSON_Processor.global_distance = distance_average
        sift_point_mask = self.get_SIFT_point_mask(static_pixels)
        return static_pixels, new_ui, sift_point_mask
    def get_static_objects(self, across_n_frames=10):
        """
        Input: greyscale Frame
        Output: Appends the sift keypoints stationary across frames to the array static objects
        """
        
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
        if len(static_points)==0:
            new_ui = True
        else:
            new_ui = False

        self.static_objects.append(static_points)
        self.update_stats(good_points, static_points)
        return static_points, new_ui

    def process_static_common_points(self, static_points):
        if len(static_points)>1:
            static_points = static_points.values()
            static_points = list(static_points)
            counts = []
            points = []
            for pt, count,size in static_points:
                counts.append(count)
                points.append(pt)
            counts = np.array(counts)
            points = np.array(points)
            indices = np.argwhere(counts == counts.max())
            static_points = np.array(static_points)
            static_points = static_points[indices]
            static_points = static_points.squeeze()
            if static_points.shape[1]==3:
                static_points = static_points[:,[0,2]]
            else:
                return []
            # static_points = points[indices]
            # static_points = static_points.squeeze()
        return static_points

    def update_stats(self, good_points, static_points):
        current_frame_total_pts = len(self.data[-1][1])
        total_pts = len(good_points)
        static_pts = len(static_points)
        dynamic_pts = total_pts - static_pts
        self.statistics.append([current_frame_total_pts, total_pts, static_pts, dynamic_pts])
        # if self.loaded_frames % 20 == 0:
        #     p = pd.DataFrame(self.statistics, columns=['current_frame', 'total_common', 'static', 'dynamic'])
        #     p.plot();
        if self.config.log_info > 2:
            print(f'{len(static_points)} SIFT points with static ratio {static_pts/total_pts}.')
        return