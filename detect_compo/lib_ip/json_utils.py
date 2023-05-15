import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


# from config.CONFIG import Configuration
# config = Configuration()

class Json_Utils:
    def __init__(self):
        self.all_frames = []
        self.current_frame = {}
        self.processed_frames = 0
        self.ui_id = 0
        self.components_in_current_frame = []

    def add_sift_statistics_to_current_frame(self, sift_statistics):
        self.current_frame['SIFT_Statistics'] = sift_statistics

    def add_component_filtration_statistics_to_current_frame(self, component_filtration_statistics):
        self.current_frame['Component_Filtration_Statistics'] = component_filtration_statistics

    def add_database_statistics_to_current_frame(self, database_statistics):
        self.current_frame['Database_Statistics'] = database_statistics[0:2]
        self.components_in_current_frame = database_statistics[2]

    def dump_current_json_to_file(self, config):

        video_name = 'video_' + str(config.input_video)
        video_name = video_name.replace('/', '_')
        video_name = video_name.replace('.mp4', '/')
        if not os.path.exists(video_name):
            os.mkdir(video_name)
        video_name = video_name.replace('/', '/detections.json')
        output = {video_name: self.all_frames}
        json_output_file_path = video_name
        # if os.path.exists(json_output_file_path):
        #     append_write = 'a'  # append if already exists
        # else:
        #     append_write = 'w'  # make a new file if not
        os.makedirs(config.output_json_folder, exist_ok=True)
        with open(json_output_file_path, 'w+') as f_out:
            json.dump(output, f_out, indent=4)

    def produce_json_for_frame_detections(self, components, frame_number, config):

        if components is None:
            img_shape = (0, 0)

        elif len(components) == 0:
            img_shape = (0, 0)
        else:
            img_shape = components[0].image_shape

        name = 'Frame ' + str(frame_number)
        output = {name: []}

        output[name].append(
            {'json_format_version': 0.1, 'id': 0, 'class': 'Background', 'frequency': 0, 'column_min': 0,
             'row_min': 0, 'column_max': img_shape[1],
             'row_max': img_shape[0], 'width': img_shape[1], 'height': img_shape[0]})

        for compo in components:
            c = {'id': compo.id, 'class': compo.category}
            c['frequency'] = (config.frame_buffer_size * len(compo.detected_in_frames))
            (c['column_min'], c['row_min'], c['column_max'], c['row_max']) = compo.bbox.put_bbox()
            c['width'] = compo.width
            c['height'] = compo.height
            c['UI_ID'] = self.ui_id
            c['frame_component_occurs_in'] = compo.detected_in_frames
            if compo.category == 'Text':
                c['word_width'] = compo.word_width
                c['content'] = compo.content
                c['confidence'] = compo.confidence


        c['sift_statistics'] = self.current_frame['SIFT_Statistics']
        c['component_filtration_statistics'] = self.current_frame['Component_Filtration_Statistics']
        c['database_statistics'] = self.current_frame['Database_Statistics']
        output[name].append(c)
        self.all_frames.append(output)
        self.processed_frames += 1
        self.dump_current_json_to_file(config)
