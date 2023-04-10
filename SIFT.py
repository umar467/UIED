#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 21:59:41 2023

@author: mf00963
"""

import numpy as np
import cv2
import datetime


def resize_by_height(org, resize_height):
    w_h_ratio = org.shape[1] / org.shape[0]
    resize_w = resize_height * w_h_ratio
    re = cv2.resize(org, (int(resize_w), int(resize_height)))
    return re


video = cv2. VideoCapture("data/input/videos/11.mp4")
ret, video_frame = video.read()
sift = cv2.SIFT_create()

while(video.isOpened()):
    if ret == True:
        video_frame = resize_by_height(video_frame, 800)
        gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        detected = cv2.drawKeypoints(gray, kp, video_frame)
        #cv2.imshow("SIFT1", detected)
        
       
        
        ret, video_frame2 = video.read()
        video_frame2 = resize_by_height(video_frame2, 800)
        gray2 = cv2.cvtColor(video_frame2, cv2.COLOR_BGR2GRAY)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        detected2 = cv2.drawKeypoints(gray2, kp2, video_frame2)
        #cv2.imshow("SIFT2", detected2)
        
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        
        matches = flann.knnMatch(des,des2,k=2)
        
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                
                pt1 = kp[m.queryIdx].pt
                pt2 = kp2[m.trainIdx].pt
                dis = cv2.norm(pt1,pt2)
                if dis<0.05:
                    matchesMask[i]=[1,0]    
                    print(i, pt1,pt2, dis)
        
        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask,
                           flags = 0)
        
        img3 = cv2.drawMatchesKnn(video_frame,kp,video_frame2,kp2,matches,None,**draw_params)
        cv2.imshow('Matches', img3)

        
        # fast = cv2.FastFeatureDetector_create()
        # fast.setNonmaxSuppression(False)
        # kp = fast.detect(gray, None)
        # kp_img = cv2.drawKeypoints(video_frame, kp, None, color=(0, 255, 0))
        # cv2.imshow('FAST', kp_img)
        
        # orb = cv2.ORB_create(nfeatures=2000)
        # kp, des = orb.detectAndCompute(gray, None)
        # kp_img = cv2.drawKeypoints(video_frame, kp, None, color=(0, 255, 0), flags=0)
        # cv2.imshow('ORB', kp_img)
       

    ret, video_frame = video.read()
    key = cv2.waitKey(1)
    if key == 113:
        break

video.release()
cv2.destroyAllWindows()
