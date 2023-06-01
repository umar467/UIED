#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 22:40:50 2023

@author: umar
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from paddleocr import PaddleOCR, draw_ocr
from detect_text import Text as text_component_builder

class text_extractor:
    
    def __init__(self, config):
        self.config = config
        self.data = None
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

    def convert_detected_text_to_components(self, detected_text, frame):
        text_components = []
        for detection in detected_text:
            for cluster in detection:
                content = cluster[1][0]
                confidence = cluster[1][1]
                id = -1
                location = {}
                location['left'] = cluster[0][0][0]
                location['right'] = cluster[0][1][0]
                location['top'] = cluster[0][2][1]
                location['bottom'] = cluster[0][0][1]

                bbox = [location['left'], location['bottom'], location['right'], location['top']]
                bbox = [int(x) for x in bbox]

                imcrop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                text_element = text_component_builder.Text(id, content, location, confidence, imcrop)
                if confidence > self.config.min_text_confidence:
                    text_components.append(text_element)
        # text_components = self.compo_filter(text_components, self.config)
        return text_components

    def compo_filter(self, compos, config):
        min_area = config.min_object_area
        C = config
        compos_new = []
        for compo in compos:
            if compo.height * compo.width < min_area:
                continue
            ratio_h = compo.width / compo.height
            ratio_w = compo.height / compo.width
            if ratio_h > C.maximum_height_ratio or ratio_w > C.maximum_width_ratio or \
                    (min(compo.height, compo.width) < C.minimum_component_height or max(ratio_h,
                                                                                         ratio_w) > C.maximum_component_ratio):
                continue
            compos_new.append(compo)
        return compos_new
    def detect_text_from_frame(self, frame):
        result = self.ocr.ocr(frame, cls=True)
        # for idx in range(len(result)):
        #     res = result[idx]
        #     for line in res:
        #         print(line)
        #self.show_detected_text_from_frame(result, frame)
        result = self.convert_detected_text_to_components(result, frame)
        return result
    def show_detected_text_from_frame(self, result, frame):
        # draw result
        from PIL import Image
        result = result[0]
        # image = Image.new('RGB', frame.shape[0:2])
        image = Image.fromarray(frame)
        #image = Image.open().convert('RGB')
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        from PIL import ImageFont
        font = ImageFont.load_default()
        im_show = draw_ocr(image, boxes, txts, scores, font_path='simfang.ttf')
        im_show = Image.fromarray(im_show)
        im_show.save('result.jpg')
        im_show.show()
        