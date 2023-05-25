import cv2
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
    config.output_folder = config.output_folder + '/' + config.input_video.split('/')[-1].split('.')[0] + '/'
    if not os.path.exists(config.output_folder):
        os.makedirs(config.output_folder)
    else:
        print("WARNING: The output folder already exists. This means that the video has already been processed.")
    video_reader_object = video_reader(config)
    start_head_location = 500
    video_reader_object.skip_frames(start_head_location)
    # video_reader_object.total_number_of_rgb_frames = 820
    SIFT_processor = SIFT_bundle(config)
    Compo_DB = Component_Database()
    JSON_Processor = json_processor(config)
    Text_Extractor = Text_Processor(config)
    component_fill_accumulator = None

    def progress(info: str):
        if config.progress_callback:
            config.progress_callback(info)
    analyzer = analyzer_class()
    pbar = tqdm(total=video_reader_object.total_number_of_rgb_frames,  desc='First pass')
    while(video_reader_object.has_enough_frames()):
        # print(video_reader_object.current_rgb_frame_number)
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
        detected_components = Compo_DB.compare_with_previously_detected_components(detected_components, frame_number,
                                                                                   current_frame_grey, JSON_Processor, config)
        JSON_Processor.process_frame()
        # visualizer.visualize_components(current_frame_rgb, detected_components, rgb=True, show=True, fill=False)
        analyzer.analyze(detected_components, current_frame_rgb, video_reader_object.total_number_of_rgb_frames, Compo_DB.compos.copy(), config)

        # JSON_Processor.process_frame()
        pbar.update(10)  # as every 10th frame processed
    video_reader_object.set_reader_head_to_frame_number(start_head_location)
    pbar.close()

    progress('second pass')
    pbar = tqdm(total=video_reader_object.total_number_of_rgb_frames,  desc='Second pass')
    while (video_reader_object.has_enough_frames()):
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
        detected_components = Compo_DB.compare_with_previously_detected_components(detected_components,
                                                                                   frame_number,
                                                                                   current_frame_grey,
                                                                                   JSON_Processor, config)

        component_image = visualizer.visualize_components(current_frame_grey, detected_components, rgb=False, show=False, fill=True)
        if component_fill_accumulator is None:
            component_fill_accumulator = np.array(component_image)
        if component_image is not None:
            component_fill_accumulator = np.dstack((component_fill_accumulator, component_image))


        if not os.path.exists(config.output_folder):
            os.makedirs(config.output_folder)

        JSON_Processor.write_json_to_file(Compo_DB)


        visualizer.Save_plots_and_heatmpas(JSON_Processor, component_fill_accumulator, config)
        visualizer.visualize_component_histograms(current_frame_rgb, detected_components, config)
        detection_frame = visualizer.visualize_components(current_frame_rgb, detected_components, rgb=True, show=False, fill=False)
        # analyzer.analyze_show(detected_components, current_frame_rgb, video_reader_object.total_number_of_rgb_frames, Compo_DB.compos.copy(), config, detection_frame)
        pbar.update(10)

    pbar.close()
