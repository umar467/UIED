import mkl
mkl.set_num_threads(1)
import cv2
cv2.setNumThreads(1)
import numpy as np
from time import time

from tqdm import tqdm

import detect_compo.lib_ip.ip_preprocessing as pre
from detect_compo.lib_ip.video_utils import video_reader
import detect_compo.lib_ip.ip_detection as det
import detect_compo.lib_ip.visualize_util as visualizer
from detect_compo.lib_ip.SIFT_utils import SIFT_Processor as SIFT_bundle
from detect_compo.lib_ip.compo_database import Compo_Database as Component_Database
from detect_compo.lib_ip.json_utils import Json_Utils as json_processor
from detect_compo.lib_ip.ocr_utils import text_extractor as Text_Processor
import os

from analyzer import Analyzer as analyzer_class
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()

def process_video(config):
    if not os.path.exists(config.output_folder):
        os.makedirs(config.output_folder)
    config.output_folder = config.output_folder + os.sep + config.input_video.split(os.sep)[-1].split('.')[0] + os.sep
    if not os.path.exists(config.output_folder):
        os.makedirs(config.output_folder)
    else:
        print("WARNING: The output folder already exists. This means that the video has already been processed.")
    '''
    The start Head location is the index of the first frame that processing will start from. If there are more than a 100 frames left afterwards otherwise the processing starts from 0.
    The max_frames is the maximum number of frames that will be processed, if the this number is greater than the total number of frames in the video, then the total number of frames will be processed.
    So in this case below, the frames 100 to 800 will be processed.
    '''
    start_head_location = 0
    max_frames = 800
    video_reader_object = video_reader(config)
    if video_reader_object.total_number_of_rgb_frames < start_head_location + 100:
        start_head_location = 0
        if video_reader_object.total_number_of_rgb_frames < 50:
            print("The video has less than 50 frames. Warning.")
    video_reader_object.skip_frames(start_head_location) # skipping the n-frames from the start
    if video_reader_object.total_number_of_rgb_frames > max_frames:
        video_reader_object.total_number_of_rgb_frames = max_frames
    SIFT_processor = SIFT_bundle(config)
    Compo_DB = Component_Database()
    JSON_Processor = json_processor(config)
    Text_Extractor = Text_Processor(config)

    def progress(info: str):
        if config.progress_callback:
            config.progress_callback(info)
    analyzer = analyzer_class(config)
    pbar = tqdm(total=video_reader_object.total_number_of_rgb_frames,  desc='First pass')

    while(video_reader_object.has_enough_frames()):

        current_frame_buffer_rgb = video_reader_object.get_Frames()
        current_frame_buffer_grey = pre.conver_frames_to_grey(current_frame_buffer_rgb)
        current_frame_buffer_gradients = pre.conver_frames_to_gradient(current_frame_buffer_grey)
        common_gradients = pre.extract_common_gradients(current_frame_buffer_gradients)
        static_pixels, new_ui = SIFT_processor.get_static_pixels(current_frame_buffer_grey, JSON_Processor)
        binary_image = pre.convert_frame_to_binary(common_gradients)
        current_frame_rgb = current_frame_buffer_rgb[-1]
        current_frame_grey = current_frame_buffer_grey[-1]
        frame_number = video_reader_object.current_rgb_frame_number
        text_components = Text_Extractor.detect_text_from_frame(current_frame_rgb)
        non_text_components = det.detect_components_from_binary_image(binary_image, static_pixels, JSON_Processor, detected_text_components=text_components, rgb_frame = current_frame_rgb)
        components = text_components + non_text_components
        detected_components = Compo_DB.compare_with_previously_detected_components(components, frame_number,
                                                                                   current_frame_grey, JSON_Processor, config,
                                                                                   force_check_previous_componenets=True)

        if new_ui:
            visualizer.new_ui_save(current_frame_buffer_rgb[0], video_reader_object.get_Frames()[-1], config)

        pbar.update(10)
        JSON_Processor.next_frame()
        visualizer.Save_plots_and_heatmpas(JSON_Processor, Compo_DB.compos.copy(), current_frame_grey, config)
        JSON_Processor.write_json_to_file(Compo_DB)
        detection_frame = visualizer.visualize_components(current_frame_rgb, detected_components, rgb=True, show=False,
                                                          fill=False)
        analyzer.analyze_show(detected_components, current_frame_rgb,
                                                video_reader_object.total_number_of_rgb_frames, Compo_DB.compos.copy(),
                                                config, detection_frame)

    pbar.close()
