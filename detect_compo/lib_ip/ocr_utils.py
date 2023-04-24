#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 22:40:50 2023

@author: umar
"""

class text_extractor:
    
    def __init__(self, config):
        self.config = config
        self.data = []
        self.current_frame = []
        self.loaded_frames = 0
        self.load_model()
        
    def load_model(self):
        import detect_text_east.ocr_east as ocr
        import detect_text_east.lib_east.eval as eval
        models = eval.load()
        print()
        