import cv2
import numpy as np
import os
import json
from time import time
import detect_compo.lib_ip.ip_preprocessing as pre
from detect_compo.lib_ip.video_utils import video_reader
import detect_compo.lib_ip.ip_detection as det
import detect_compo.lib_ip.visualize_util as visualizer
from detect_compo.lib_ip.SIFT_utils import SIFT_Processor as SIFT_bundle
from detect_compo.lib_ip.compo_database import Compo_Database as Component_Database
from detect_compo.lib_ip.json_utils import Json_Utils as json_processor
from detect_compo.lib_ip.ocr_utils import text_extractor as Text_Processor

def process_video(config):
    video_reader_object = video_reader(config)
    #video_reader_object.skip_frames(video_reader_object.total_number_of_rgb_frames-140)
    SIFT_processor = SIFT_bundle()
    Compo_DB = Component_Database()
    Compo_DB_Text = Component_Database()
    JSON_Processor = json_processor()
    Text_Extractor = Text_Processor(config)
    while(video_reader_object.has_enough_frames()):
        start_time = time()
        current_frame_buffer_rgb  = video_reader_object.get_Frames()
        current_frame_buffer_grey = pre.conver_frames_to_grey(current_frame_buffer_rgb)
        current_frame_buffer_gradients = pre.conver_frames_to_gradient(current_frame_buffer_grey)
        common_gradients = pre.extract_common_gradients(current_frame_buffer_gradients)
        binary_image = pre.convert_frame_to_binary(common_gradients)

        static_pixels = SIFT_processor.get_static_pixels(current_frame_buffer_grey)
        detected_text_components = Text_Extractor.detect_text_from_frame(current_frame_buffer_rgb[-1])
        detected_components = det.detect_components_from_binary_image(binary_image, static_pixels=static_pixels, detected_text_components = detected_text_components)
        db_start_time = time()
        detected_components = Compo_DB.compare_with_previously_detected_components(detected_components, video_reader_object.current_rgb_frame_number)
        detected_text_components = Compo_DB_Text.compare_with_previously_detected_components(detected_text_components, video_reader_object.current_rgb_frame_number)
        JSON_Processor.produce_json_for_frame_detections(detected_components, detected_text_components, video_reader_object.current_rgb_frame_number, config)

        print(f'Start time {db_start_time-start_time} DB time {time()-db_start_time}')

        components_result_image = visualizer.visualize_components(current_frame_buffer_grey[-1], detected_components, show=False, rgb=False)
        all_components_result_image = visualizer.visualize_components(components_result_image, detected_text_components,
                                                                  show=False, rgb=True)
        components_with_static_pixels_result_image = visualizer.visualize_points(all_components_result_image, static_pixels, show=False)

        final_result_to_show = np.hstack([components_with_static_pixels_result_image, common_gradients, binary_image])

        cv2.imshow('', final_result_to_show)
        cv2.waitKey(10)

    SIFT_processor.plot_SIFT_detection_plots()
