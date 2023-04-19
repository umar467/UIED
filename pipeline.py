#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 18:17:24 2023

@author: umar
"""

import cv2
import numpy as np
import detect_compo.lib_ip.ip_draw as draw
import detect_compo.lib_ip.ip_preprocessing as pre
from detect_compo.lib_ip.Component import Component
import detect_compo.lib_ip.Component as Compo
from config.NEW_CONFIG_UIED import Configuration
config = Configuration()
from detect_compo.lib_ip.video_utils import video_reader
from detect_compo.lib_ip.SIFT_utils import SIFT_Processor
from detect_compo.lib_ip.component_detection_utils import component_detector
import detect_compo.lib_ip.visualize_util as visualizer

video  = video_reader(config)
frame = video.get_processed_frame()

sift_processor = SIFT_Processor(config)
sift_processor.get_static_objects(frame)
        
frame2 = video.get_processed_frame()
sps = sift_processor.get_static_objects(frame2)


component_detector = component_detector(config)
detected_compos = component_detector.get_components(frame)

visualizer.visualize_points(frame, sps)


"""
import numpy as np
import cv2
from os.path import join as pjoin
from pathlib import Path
import datetime
import detect_compo.lib_ip.ip_detection as det
import detect_compo.lib_ip.ip_preprocessing as pre
import detect_compo.lib_ip.ip_draw as draw



def resize_by_height(org, resize_height):
    w_h_ratio = org.shape[1] / org.shape[0]
    resize_w = resize_height * w_h_ratio
    re = cv2.resize(org, (int(resize_w), int(resize_height)))
    return re
def normalize_fg(fg_avg):
    mean = fg_avg.mean()
    high = fg_avg.max()
    p98 = high - ((high-mean)*0.8)
    fg_avg[fg_avg<p98]=0
    fg_avg[fg_avg>p98]=255
    fg_avg[fg_avg==p98]=0
    return fg_avg

def process_frame(frame):
    #frame = cv2.medianBlur(frame, 10)
    bin_frame = pre.binarization(frame, 20)
    det.rm_line(bin_frame)
    uicompos = det.component_detection(bin_frame, min_obj_area=5)
    return uicompos

out_path = 'data/output/frames/11_SIFT_merged/'
Path(out_path).mkdir(parents=True, exist_ok=True)
frame_size =800
video = cv2. VideoCapture("data/input/videos/11.mp4")
ret, video_frame = video.read()
sift = cv2.SIFT_create()
video_frame = resize_by_height(video_frame, frame_size)
outline = np.zeros(video_frame.shape[0:2])
outline_historical = outline.copy()
outline_empty = outline.copy()
counter = 0

while(video.isOpened()):
    if ret == True:
        video_frame = resize_by_height(video_frame, frame_size)
        gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        detected = cv2.drawKeypoints(gray, kp, video_frame)
        #cv2.imshow("SIFT1", detected)
        
       
        
        ret, video_frame2 = video.read()
        video_frame2 = resize_by_height(video_frame2, frame_size)
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
                    #print(i, pt1,pt2, dis)
                    cv2.circle(outline, (int(pt1[0]),int(pt1[1])), 5, (255,255,255), thickness=-1, lineType=8, shift=0)
        
        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask,
                           flags = 0)
        
        img3 = cv2.drawMatchesKnn(video_frame,kp,video_frame2,kp2,matches,None,**draw_params)
        cv2.imshow('Matches', resize_by_height(img3,800))
        cv2.imshow('outline', resize_by_height(outline,800))
        counter = counter + 1
        if counter%6==0:
            outline_historical = outline_historical + outline
            outline = outline_empty.copy()
            outline_historical = normalize_fg(outline_historical)
            cv2.imshow('outline_historical', resize_by_height(outline_historical,800))
            uicompos = process_frame(gray2)
            drawn = draw.draw_bounding_box(video_frame2, uicompos, show=False, name='components_SIFT', wait_key=1,SIFT_average_image = outline_historical)
            fno = str(counter).zfill(5)
            fname = out_path+fno+'.jpg'
            cv2.imshow('Processed', resize_by_height(drawn,800))
            #cv2.imwrite(fname,drawn)
            #cv2.imwrite(out_path+str(counter)+'.jpg',drawn)
            #draw.draw_bounding_box(video_frame2, uicompos, show=True, name='components', wait_key=1)
            
        else:
            outline_historical = outline_historical + outline
            outline = outline_empty.copy()
            uicompos = process_frame(gray2)
            drawn = draw.draw_bounding_box(video_frame2, uicompos, show=False, name='components_SIFT', wait_key=1,SIFT_average_image = outline_historical)
            fno = str(counter).zfill(5)
            fname = out_path+fno+'.jpg'
            cv2.imshow('Processed', resize_by_height(drawn,800))
            #cv2.imwrite(fname,drawn)

    ret, video_frame = video.read()
    key = cv2.waitKey(1)
    if key == 113:
        break

video.release()
cv2.destroyAllWindows()
print("Finsihed!!!")

"""