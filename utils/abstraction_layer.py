import cv2
import numpy as np
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
    SIFT_processor = SIFT_bundle(config)
    Compo_DB = Component_Database()
    JSON_Processor = json_processor()
    Text_Extractor = Text_Processor(config)
    while(video_reader_object.has_enough_frames()):
        current_frame_buffer_rgb = video_reader_object.get_Frames()
        current_frame_buffer_grey = pre.conver_frames_to_grey(current_frame_buffer_rgb)
        current_frame_buffer_gradients = pre.conver_frames_to_gradient(current_frame_buffer_grey)
        common_gradients = pre.extract_common_gradients(current_frame_buffer_gradients)
        static_pixels = SIFT_processor.get_static_pixels(current_frame_buffer_grey, JSON_Processor)
        binary_image = pre.convert_frame_to_binary(common_gradients)
        current_frame_rgb = current_frame_buffer_rgb[-1]
        current_frame_grey = current_frame_buffer_grey[-1]
        frame_number = video_reader_object.current_rgb_frame_number
        detected_components = det.detect_components_from_binary_image(binary_image, static_pixels, JSON_Processor)
        detected_components += Text_Extractor.detect_text_from_frame(current_frame_rgb)

        Compo_DB.compare_with_previously_detected_components(detected_components, frame_number, current_frame_grey, JSON_Processor)
        JSON_Processor.produce_json_for_frame_detections(detected_components, frame_number, config)