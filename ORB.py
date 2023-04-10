#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 23:33:11 2023

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

orb = cv2.ORB_create(nfeatures=2000)

while(video.isOpened()):
    if ret == True:
        video_frame = resize_by_height(video_frame, 800)
        gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        
        # for i in range(50):
        #     video.read()
        
        ret, video_frame2 = video.read()
        video_frame2 = resize_by_height(video_frame2, 800)
        gray2 = cv2.cvtColor(video_frame2, cv2.COLOR_BGR2GRAY)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        bf = cv2.BFMatcher_create(cv2.NORM_HAMMING,crossCheck=True)     
        matches = bf.match(des, des2)
        matches = sorted(matches,key=lambda x:x.distance)
        new_matches = []
        for i,(m) in enumerate(matches):
            pt1 = kp[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt
            dis = cv2.norm(pt1,pt2)
            if dis<0.05:
                print(i, pt1,pt2, dis)
                new_matches.append(m)
        
        matches = new_matches
        ORB_matches =cv2.drawMatches(video_frame, kp, video_frame2, kp2, matches[:500], None, flags=2)
        cv2.imshow('ORB_matches', ORB_matches)

       

    ret, video_frame = video.read()
    key = cv2.waitKey(1)
    if key == 113:
        break

video.release()
cv2.destroyAllWindows()
