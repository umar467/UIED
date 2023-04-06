#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 18:13:48 2023

@author: umar
"""

from os.path import join as pjoin
import cv2
import os
import detect_compo.lib_ip.ip_detection as det
import detect_compo.lib_ip.ip_preprocessing as pre


def resize_by_height(org, resize_height=600):
    w_h_ratio = org.shape[1] / org.shape[0]
    resize_w = resize_height * w_h_ratio
    re = cv2.resize(org, (int(resize_w), int(resize_height)))
    return re

def optical_change(path):
    import cv2
    import numpy as np
    camera = cv2.VideoCapture(path)
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,4))
    kernel = np.ones((5,5),np.uint8)
    background = None
    
    while (True):
     ret, frame = camera.read()
     frame = resize_by_height(frame)

     if background is None:
         background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         background = cv2.GaussianBlur(background, (21, 21), 0)
         continue
    
     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    
     diff = cv2.absdiff(background, gray_frame)
     cv2.imshow('grad',diff)
     diff = cv2.adaptiveThreshold(diff,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)#cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
     cv2.imshow('diff',diff)
     #diff = cv2.dilate(diff, es, iterations = 2)
     cnts, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
     for c in cnts:
         if cv2.contourArea(c) < 1500:
             continue
         (x, y, w, h) = cv2.boundingRect(c)
         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
     cv2.imshow("contours", frame)
     cv2.imshow("dif", diff)
     cv2.waitKey(10)
     scene_change = cv2.absdiff(background, gray_frame).sum()
     print(scene_change)
     if scene_change > 5000000:
         background = gray_frame
         print("------------>>>>>>>")
     
    cv2.destroyAllWindows()
    camera.release()
if __name__ == '__main__':
    optical_change("data/input/videos/4.mp4")