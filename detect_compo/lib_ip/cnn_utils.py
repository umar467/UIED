#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 10:26:32 2023

@author: umar
"""


class cnn:
    
    def __init__(self, config):
        self.config = config
        self.data = []
        self.current_frame = []
        self.loaded_frames = 0
        from cnn.CNN import CNN
        self.model = CNN('Elements', is_load=True)
        
        
    def process(self, frame, compos):
        img = self.model.preprocess_img(frame[1])
        self.model.predict(img, compos)
        