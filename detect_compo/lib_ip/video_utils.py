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
        self.video = cv2.VideoCapture(self.video_path)
        self.current_rgb_frame_number = 0
        self.total_number_of_rgb_frames = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        if self.config.log_info:
            self.print_stats()

    def has_next(self):
        if self.current_rgb_frame_number < (self.total_number_of_rgb_frames + 2):
            return True
        return False
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
        frame_numbers = []
        for _ in range(no_of_frames):
            frame = self.get_processed_frame()
            if frame is not None:
                frames.append(frame)
                frame_numbers.append(self.current_rgb_frame_number)
        frames = np.array(frames)
        return frames, frame_numbers

    def get_all_Frames(self):
        no_of_frames = self.total_number_of_rgb_frames
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 1)
        frames = []
        for _ in range(no_of_frames):
            frame = self.get_processed_frame()
            if frame is not None:
                frames.append(frame)
        frames = np.array(frames)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.current_rgb_frame_number - 1)
        return frames


    def get_processed_frame(self, rgb_frame = None, downsampling = True):
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
            if downsampling:
                rgb_frame = image_processing.resize_by_height(rgb_frame, self.config.resize_input_image_height)
            else:
                rgb_frame = image_processing.resize_by_height(rgb_frame, rgb_frame.shape[0])
        #binary_rgb_frame, grey_frame = image_processing.binarization(rgb_frame, self.config.grad_min, self.config.morphology_size)

        # blur = cv2.bilateralFilter(rgb_t, 9, 75, 75)

        # cv2.imshow('frame<<bef', rgb_frame)
        # cv2.waitKey(0)
        # rgb_frame = rgb_frame>>3
        # rgb_frame = rgb_frame<<3
        # cv2.imshow('frame>>', rgb_frame)
        # cv2.waitKey(0)
        return rgb_frame
    
    def get_next_frame(self):
        '''
        Input: Object's internal frame read head'
        Output: Returns the RGB frame next to the current read head
        '''
        ret, rgb_frame = self.video.read()
        if ret:
            self.current_rgb_frame_number+=1

            # rgb_frame = cv2.flip(rgb_frame,-1)

            # rgb_frame = cv2.putText(rgb_frame, f'Im Readable!', (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
            # rgb_frame = cv2.putText(rgb_frame, f'Im Not!', (100, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
            # rgb_frame = cv2.putText(rgb_frame, f'How about this!', (100, 1150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 255, 0), 5)
            # rgb_frame = cv2.putText(rgb_frame, f'Definitely not me!', (500, 1150), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
            #                         (255, 0, 255), 5)
            # rgb_frame = cv2.putText(rgb_frame, f'Me NOT me!', (200, 1550), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            #                         (255, 0, 0), 5)
            # rgb_frame = cv2.putText(rgb_frame, f'YES me!', (450, 1850), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            #                         (0, 0, 255), 5)
            # # read image logo.png from the desktop
            # logo = cv2.imread('logo.png' ,cv2.IMREAD_UNCHANGED)
            # logo2 = cv2.imread('logo2.png',cv2.IMREAD_UNCHANGED)
            # logo3 = cv2.imread('logo4.png',cv2.IMREAD_UNCHANGED)
            # #resize the image to 200x200
            # logo = cv2.resize(logo, (200, 200))
            # logo2 = cv2.resize(logo2, (200, 200))
            # logo3 = cv2.resize(logo3, (200, 200))
            # overlay = logo3
            # background = rgb_frame
            #
            # height, width = overlay.shape[:2]
            # for y in range(height):
            #     for x in range(width):
            #         for offset in [(400, 600), (800, 300), (580, 600)]:
            #
            #             overlay_color = overlay[y, x, :3]  # first three elements are color (RGB)
            #             overlay_alpha = overlay[
            #                                 y, x, 3] / 255  # 4th element is the alpha channel, convert from 0-255 to 0.0-1.0
            #
            #             yb = y + offset[0]
            #             xb = x + offset[1]
            #
            #             # get the color from the background image
            #             background_color = background[yb, xb]
            #
            #             # combine the background color and the overlay color weighted by alpha
            #             composite_color = background_color * (1 - overlay_alpha) + overlay_color * overlay_alpha
            #
            #             # update the background image in place
            #             background[yb, xb] = composite_color

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

    def get_neighbours_of_specific_frame(self, requested_rgb_frame_number, no_of_neighbours, downsampling = True):
        '''
        Input: Takes a specific frame number as input
        Output: Returns the N frames before and after the requested frame
        '''
        try:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, requested_rgb_frame_number-no_of_neighbours)
            frames = []
            for _ in range(no_of_neighbours*2):
                rgb_frame  = self.get_processed_frame(downsampling=downsampling)
                frames.append(rgb_frame)
            self.video.set(cv2.CAP_PROP_POS_FRAMES, self.current_rgb_frame_number-1)
            return frames
        except Exception as e:
            print(f'rgb_frame Number Exceeds Video Length of {self.total_number_of_rgb_frames} rgb_frames !!')
            return None


    def get_specific_frame(self, requested_rgb_frame_number, downsampling = True):
        '''
        Input: Takes a specific frame number as input
        Output: Returns the proper sized RGB_frame and its greyscale image
        '''
        if requested_rgb_frame_number < self.total_number_of_rgb_frames:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, requested_rgb_frame_number-1)
            rgb_frame  = self.get_processed_frame(downsampling=downsampling)
            self.video.set(cv2.CAP_PROP_POS_FRAMES, self.current_rgb_frame_number-1)
            return rgb_frame #self.get_processed_frame(rgb_frame)
        else:
            print(f'rgb_frame Number Exceeds Video Length of {self.total_number_of_rgb_frames} rgb_frames !!')
            return None