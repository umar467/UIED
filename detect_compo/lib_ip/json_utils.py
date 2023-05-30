import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


# from config.CONFIG import Configuration
# config = Configuration()

class Json_Utils:
    def __init__(self, config):
        self.all_frames = []
        self.current_frame = {}
        self.processed_frames = 0
        self.ui_id = 0
        self.components_in_current_frame = []
        self.config = config

    def add_sift_statistics_to_current_frame(self, sift_statistics):
        self.current_frame['SIFT_Statistics'] = sift_statistics

    def add_component_filtration_statistics_to_current_frame(self, component_filtration_statistics):
        self.current_frame['Component_Filtration_Statistics'] = component_filtration_statistics

    def add_database_statistics_to_current_frame(self, database_statistics):
        self.current_frame['Database_Statistics'] = database_statistics[0:2]
        self.components_in_current_frame = database_statistics[2]

    def process_frame(self):
        self.all_frames.append(self.current_frame)
        self.processed_frames += 1
        self.current_frame = {}

    def get_stats(self):
        f = []
        d = []
        s = []
        for frame in self.all_frames:
            f.append(frame['Component_Filtration_Statistics'])
            d.append(frame['Database_Statistics'])
            s.append(frame['SIFT_Statistics'])

        return [f, d, s]

    def visualize_sift(self):
        sift_stats = []
        for frame in self.all_frames:
            if 'SIFT_Statistics' in frame:
                sift_stats.append(frame['SIFT_Statistics'])
        self.save_plots(sift_stats)

    def save_plots(self, data):
        p = pd.DataFrame(data, columns=['current_frame', 'total_common', 'static', 'dynamic'])
        plot = p.plot();
        # plot.title('SIFT Features across Frames')
        plot.set_xlabel("Frames x 10")
        plot.set_ylabel("Frequency")
        fig = plot.get_figure()
        fig.savefig("sift.png")
        plt.close()

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
        os.makedirs(config.output_folder, exist_ok=True)
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
            c['bbox_historical'] = compo.bbox_historical
            c['frame_component_occurs_in'] = compo.detected_in_frames
            assert len(compo.detected_in_frames) == len(compo.bbox_historical)
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

    def produce_json_for_component(self, component):
        c = {'id': component.id, 'class': component.category}
        c['frequency'] = 1
        (c['column_min'], c['row_min'], c['column_max'], c['row_max']) = component.bbox.put_bbox()
        c['width'] = component.width
        c['height'] = component.height
        c['UI_ID'] = self.ui_id
        c['bbox_historical'] = component.bbox_historical
        c['frame_component_occurs_in'] = component.detected_in_frames
        assert len(component.detected_in_frames) == len(component.bbox_historical)
        if component.category == 'Text':
            c['word_width'] = component.word_width
            c['content'] = component.content
            c['confidence'] = component.confidence
        #else:
            #c['image_crop'] = np.zeros((128, 128)).tolist()
        return c

    def produce_json_from_database_components(self, database):
        components = database.get_all_components()
        json_format_version = .3
        json_output = {'json_format_version': json_format_version, 'elements': [], 'warnings': []}
        sample_warning = {'warning_type': 'Elements Too Close Warning', 'bbox': [22, 55, 66, 77], 'frames_occurs_in': [1, 2, 3]}
        json_output['warnings'].append(sample_warning)
        for component in components:
            component_json = self.produce_json_for_component(component)
            json_output['elements'].append(component_json)
        return json_output

    def write_json_to_file(self, database):
        json_output = self.produce_json_from_database_components(database)
        # json_output_file_path = "test.json"
        json_output_file_path = self.config.output_folder + "/detections.json"
        with open(json_output_file_path, 'w+') as f_out:
            json.dump(json_output, f_out, indent=4)
