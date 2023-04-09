#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 18:13:48 2023

@author: umar
"""

from os.path import join as pjoin
import cv2
import os
import random
import numpy as np
import detect_compo.lib_ip.ip_detection as det
import detect_compo.lib_ip.ip_preprocessing as pre
import detect_compo.lib_ip.ip_draw as draw




def optical_change(path):
    camera = cv2.VideoCapture(path)
    background = None
    frame_diff_height=900
    frame_compute_height=900
    frame_diff_treshold = 2822450
    frame_averaging = False
    diff_frame =[]
    frame=[]
    gray_frame=[]
    uicompos=[]
    keep_reading = True
    fno=0

    def resize_by_height(org, resize_height):
        w_h_ratio = org.shape[1] / org.shape[0]
        resize_w = resize_height * w_h_ratio
        re = cv2.resize(org, (int(resize_w), int(resize_height)))
        return re    
    
    def frame_diff(frame):
        diff = cv2.absdiff(background, frame)
        diff = cv2.adaptiveThreshold(diff,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        diff = pre.binarization(diff, 20)
        #cv2.imshow('freame', diff)
        #cv2.waitKey(10)
        diff = diff.sum()
        #print(diff)
        return diff
    def process_diff_frame(frame):
        frame = resize_by_height(frame, frame_diff_height)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (21, 21), 0)
        return frame
    def process_compute_frame(frame):
        frame = resize_by_height(frame,frame_compute_height)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame = cv2.medianBlur(frame, 10)
        return frame, gray_frame
    def process_frame(frame):
        
        frame, gray_frame = process_compute_frame(frame)
        
        bin_frame = pre.binarization(gray_frame, 20)
        det.rm_line(bin_frame)
        
        uicompos = det.component_detection(bin_frame, min_obj_area=5)
        if frame_averaging:
            diff_frame = process_diff_frame(frame)
            return diff_frame, frame, gray_frame, uicompos
        else:
            return frame, gray_frame, uicompos
    def find_contours(diff, frame):
        cnts, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        for c in cnts:
            if cv2.contourArea(c) < 1500:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
       
        cv2.imshow("contours", frame)
        cv2.waitKey(10)
    def random_sample_compos(fg_avg):
        totalFrames = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
        randomFrameNumbers = random.sample(range(0, totalFrames), 10)
        for x in randomFrameNumbers:
            camera.set(cv2.CAP_PROP_POS_FRAMES,x)
            _, image = camera.read()
            diff_frame, frame, gray_frame, uicompos = process_frame(image)
            fg_avg = fg_avg + draw.avgboxx(frame, uicompos)
        camera.set(cv2.CAP_PROP_POS_FRAMES,0)
        return fg_avg
    def normalize_fg(fg_avg):
        mean = fg_avg.mean()
        high = fg_avg.max()
        p98 = high - ((high-mean)*0.8)
        fg_avg[fg_avg<p98]=0
        fg_avg[fg_avg>p98]=255
        fg_avg[fg_avg==p98]=0
        fg = fg_avg.copy()
        fv_avg = 0
        return fg, fg_avg
    while (keep_reading):
     ret, frame = camera.read()
     if ret ==0 or fno>10:
         keep_reading = False
         continue
     fno= fno+1
     if frame_averaging:
         diff_frame, frame, gray_frame, uicompos =process_frame(frame)
     else:
         frame, gray_frame, uicompos = process_frame(frame)
         
          
     if background is None:
         background = diff_frame
         #fourcc = cv2.VideoWriter_fourcc(*'MP4V')
         #fourcc = cv2.VideoWriter_fourcc(*'MP4V')
         #out=cv2.VideoWriter('out.mp4', fourcc,20.0,frame.shape[0:2])
         out = cv2.VideoWriter('filename.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, frame.shape[0:2])
         if frame_averaging:
             fg = draw.avgboxx(frame, uicompos)
             fg_avg_global = random_sample_compos(fg)
             fg_avg = fg_avg_global.copy()
             fg, fg_avg = normalize_fg(fg_avg)
         continue
     if frame_averaging:
         if frame_diff(diff_frame) > frame_diff_treshold:
             background = diff_frame
             #print("Scene Cahngeed !!! -------")
             fg, fg_avg = normalize_fg(fg_avg)
             continue

     if frame_averaging:
         drawn = draw.draw_bounding_box(frame, uicompos, show=False, name='components', wait_key=1, fg =fg)
         fg_avg = fg_avg + draw.avgboxx(frame, uicompos)
     else:
         drawn = draw.draw_bounding_box(frame, uicompos, show=False, name='components', wait_key=10)
     #cv2.imshow('12',fg)
     out.write(gray_frame)
     
    
    camera.release()
    out.release()
    print("video saved")
    cv2.destroyAllWindows()
if __name__ == '__main__':
    optical_change("data/input/videos/4.mp4")