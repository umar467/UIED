#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 18:13:48 2023

@author: umar
"""

from os.path import join as pjoin
import cv2
import os
import numpy as np
import detect_compo.lib_ip.ip_detection as det
import detect_compo.lib_ip.ip_preprocessing as pre
import detect_compo.lib_ip.ip_draw as draw




def optical_change(path):
    camera = cv2.VideoCapture(path)
    background = None
    frame_diff_height=600
    frame_compute_height=600
    frame_diff_treshold = 3822450


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
        diff_frame = process_diff_frame(frame)
        frame, gray_frame = process_compute_frame(frame)
        return diff_frame, frame, gray_frame
    def find_contours(diff, frame):
        cnts, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        for c in cnts:
            if cv2.contourArea(c) < 1500:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
       
        cv2.imshow("contours", frame)
        cv2.waitKey(10)
        
    while (True):
     _, frame = camera.read()
     diff_frame, frame, gray_frame = process_frame(frame)

     bin_frame = pre.binarization(gray_frame, 20)
     det.rm_line(bin_frame)
     uicompos = det.component_detection(bin_frame, min_obj_area=5)
          
     if background is None:
         background = diff_frame
         fg = draw.avgboxx(frame, uicompos)
         fg_avg = fg.copy()
         continue
     elif frame_diff(diff_frame) > frame_diff_treshold:
         background = diff_frame
         mean = fg_avg.mean()
         high = fg_avg.max()
         p98 = mean#high - ((high-mean)*0.2)
         fg_avg[fg_avg<p98]=0
         fg_avg[fg_avg>p98]=255
         fg_avg[fg_avg==p98]=0
         fg = fg_avg.copy()
         
         continue

     draw.draw_bounding_box(fg, frame, uicompos, show=True, name='components', wait_key=10)
     fg_avg = fg_avg + draw.avgboxx(frame, uicompos)
     cv2.imshow('12',fg)
     
    cv2.destroyAllWindows()
    camera.release()
if __name__ == '__main__':
    optical_change("data/input/videos/11.mp4")