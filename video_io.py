#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 18:23:30 2023

@author: mf00963
"""
import numpy as np
import cv2
import datetime
imgH=2160
imgW=966
#FOURCC is short for “four character code” – 
#an identifier for a video codec, compression format,
#cv2.VideoWriter – Saves the output video to a directory.
fourcc = cv2.VideoWriter_fourcc(*'avc1')
video_write = cv2.VideoWriter('t.avi', fourcc, 60.0, (imgW*2, imgH), True)
#video_write=cv2.VideoWriter('video_opencv.mp4',fourcc,20,(imgW, imgH))
video = cv2. VideoCapture("data/input/videos/11.mp4")
ret, video_frame=video.read()
while(video.isOpened()):
    if ret==True:
        
        cv2.putText(video_frame, "Date: " + str( datetime.datetime.now()),
                    (50,50), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, 
                    (0, 255, 255), 
                    2)
        video_write.write(video_frame)
        cv2.imshow("Source", video_frame)
    ret, video_frame=video.read()
    key = cv2.waitKey(20)
    # if key q is pressed then break 
    if key == 113:
        break 
    
#finally destroy/close all open windows
video.release()
cv2.destroyAllWindows()