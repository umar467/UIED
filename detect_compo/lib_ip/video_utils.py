#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 18:55:29 2023

@author: umar
"""
import cv2
import numpy as np
import detect_compo.lib_ip.ip_preprocessing as image_processing
import detect_compo.lib_ip.ip_detection as component_detection

class video_reader:
    
    def __init__(self, config):
        self.config = config
        self.video_path = config.input_video
        self.video = cv2. VideoCapture(self.video_path)
        self.current_rgb_frame_number = 0
        self.total_number_of_rgb_frames = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        if self.config.log_info:
            self.print_stats()

    def has_enough_frames(self):
        if self.current_rgb_frame_number + self.config.frame_buffer_size < self.total_number_of_rgb_frames + 1:
            return True
        else:
            return False
    def print_stats(self):
        print(f'Successfully Loaded Video {self.video_path} containing {self.total_number_of_rgb_frames} rgb_frames with the pointer at rgb_frame no.{self.current_rgb_frame_number}')

    def get_Frames(self):
        no_of_frames = self.config.frame_buffer_size
        frames = []
        for _ in range(no_of_frames):
            frame = self.get_processed_frame()
            if frame is not None:
                frames.append(frame)
        frames = np.array(frames)
        return frames

    def get_processed_frame(self, rgb_frame = None):
        '''
        Input: Objects internal frame read head
        Output: Returns the proper sized rgb_frame and its greyscale image
        '''
        if rgb_frame is None:
            rgb_frame = self.get_next_frame()
            if rgb_frame is None:
                return None
        if self.config.input_frame_blur_kernel_size is not None:
            rgb_frame = cv2.medianBlur(rgb_frame, self.config.input_frame_blur_kernel_size)
        if self.config.resize_input_image_height is  not None:
            rgb_frame = image_processing.resize_by_height(rgb_frame, self.config.resize_input_image_height)
        #binary_rgb_frame, grey_frame = image_processing.binarization(rgb_frame, self.config.grad_min, self.config.morphology_size)
        return rgb_frame
    
    def get_next_frame(self):
        '''
        Input: Object's internal frame read head'
        Output: Returns the RGB frame next to the current read head
        '''
        ret, rgb_frame = self.video.read()
        if ret:
            self.current_rgb_frame_number+=1
            return rgb_frame
        else:
            print("Couldn't read rgb_frame !!")
            return None
        
    def skip_frames(self, skip_n):
        new_frame_header = self.current_rgb_frame_number+skip_n
        if new_frame_header < self.total_number_of_rgb_frames:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, new_frame_header-1)
            self.current_rgb_frame_number = new_frame_header
        else:
            print(f'rgb_frame Number Exceeds Video Length of {self.total_number_of_rgb_frames} rgb_frames !!')
            return None

    def set_reader_head_to_frame_number(self, frame_number):
        if frame_number < self.total_number_of_rgb_frames:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
            self.current_rgb_frame_number = frame_number
        else:
            print(f'rgb_frame Number Exceeds Video Length of {self.total_number_of_rgb_frames} rgb_frames !!')
            return None
    def get_specific_frame(self, requested_rgb_frame_number):
        '''
        Input: Takes a specific frame number as input
        Output: Returns the proper sized RGB_frame and its greyscale image
        '''
        if requested_rgb_frame_number < self.total_number_of_rgb_frames:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, requested_rgb_frame_number-1)
            rgb_frame  = self.get_next_rgb_frame()
            self.video.set(cv2.CAP_PROP_POS_FRAMES, self.current_rgb_frame_number-1)
            return self.get_processed_frame(rgb_frame)
        else:
            print(f'rgb_frame Number Exceeds Video Length of {self.total_number_of_rgb_frames} rgb_frames !!')
            return None