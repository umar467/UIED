import cv2
import numpy as np
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
    video_reader_object.skip_frames(140)
    SIFT_processor = SIFT_bundle(config)
    Compo_DB = Component_Database()
    Compo_DB_Text = Component_Database()
    JSON_Processor = json_processor()
    Text_Extractor = Text_Processor(config)
    Text_Statistics = []
    while(video_reader_object.has_enough_frames()):
        current_frame_buffer_rgb  = video_reader_object.get_Frames()
        current_frame_buffer_grey = pre.conver_frames_to_grey(current_frame_buffer_rgb)
        current_frame_buffer_gradients = pre.conver_frames_to_gradient(current_frame_buffer_grey)
        common_gradients = pre.extract_common_gradients(current_frame_buffer_gradients)
        binary_image = pre.convert_frame_to_binary(common_gradients)

        static_pixels = SIFT_processor.get_static_pixels(current_frame_buffer_grey)
        detected_text_components = Text_Extractor.detect_text_from_frame(current_frame_buffer_rgb[-1])
        detected_components = det.detect_components_from_binary_image(binary_image, static_pixels=static_pixels, detected_text_components = detected_text_components, Text_Statistics=Text_Statistics, config = config)
        component_cropsbb, component_crop_imagesbb = visualizer.visualize_component_crops(current_frame_buffer_grey[-1],
                                                                                      detected_components, show=False,
                                                                                      rgb=False, fill=True)
        detected_components2 = Compo_DB.compare_with_previously_detected_components(detected_components, video_reader_object.current_rgb_frame_number, current_frame_buffer_grey[-1], force_check_previous_componenets=True)
        detected_components = Compo_DB.compare_with_previously_detected_components(detected_components, video_reader_object.current_rgb_frame_number, current_frame_buffer_grey[-1], force_check_previous_componenets=False)
        detected_text_components = Compo_DB_Text.compare_with_previously_detected_components(detected_text_components,
                                                                                             video_reader_object.current_rgb_frame_number,
                                                                                             current_frame_buffer_grey[
                                                                                                 -1],
                                                                                             force_check_previous_componenets=False)
        detected_text_components2 = Compo_DB_Text.compare_with_previously_detected_components(detected_text_components, video_reader_object.current_rgb_frame_number, current_frame_buffer_grey[-1], force_check_previous_componenets=True)
        #JSON_Processor.produce_json_for_frame_detections(detected_components, detected_text_components, video_reader_object.current_rgb_frame_number, config)

        # counts = visualizer.compare_component_crops(current_frame_buffer_grey[-1], current_frame_buffer_grey[0], detected_components)
        component_crops, component_crop_images = visualizer.visualize_component_crops(current_frame_buffer_grey[-1], detected_components, show=False,rgb=False, fill=True)
        component_crops2, component_crop_images2 = visualizer.visualize_component_crops(current_frame_buffer_grey[-1],
                                                                                      detected_components2, show=False,
                                                                                      rgb=False, fill=True)
        component_text_crops, component_text_crop_images = visualizer.visualize_component_crops(current_frame_buffer_grey[-1], detected_text_components, show=False, rgb=False, fill=True)
        component_text_crops2, component_text_crop_images2 = visualizer.visualize_component_crops(
            current_frame_buffer_grey[-1], detected_text_components2, show=False, rgb=False, fill=True)
        point_image = visualizer.visualize_points(current_frame_buffer_grey[-1], static_pixels, show=False, rgb=False)
        component_crops += component_text_crops
        component_crops2 += component_text_crops2
        #component_crops += point_image
        cv2.imshow('', np.hstack([component_cropsbb, current_frame_buffer_grey[-1], component_crops, component_crops2]))
        cv2.waitKey(100)

        # if Compo_DB_Text.compute_frame_statistics(detected_text_components, video_reader_object.current_rgb_frame_number):# or Compo_DB.compute_frame_statistics(detected_components, video_reader_object.current_rgb_frame_number):
        #     JSON_Processor.ui_id = JSON_Processor.ui_id + 1
        #     component_crops, component_crop_images = visualizer.visualize_component_crops(current_frame_buffer_grey[-1], detected_components, show=False, rgb=False)
        #     component_text_crops, component_text_crop_images = visualizer.visualize_component_crops(current_frame_buffer_grey[-1], detected_text_components, show=False, rgb=False)
        #     component_crops += component_text_crops
        #
        #     final_result_to_show = np.hstack([current_frame_buffer_grey[-1], component_crops, component_crop_images, component_text_crop_images])
        #     video_name = 'video_' + str(config.input_video)
        #     video_name = video_name.replace('/', '_')
        #     video_name = video_name.replace('.mp4', '/')
        #
        #     ui_output_image_file = f'{video_name}/UI_Image_{str(JSON_Processor.ui_id)}_{str(video_reader_object.current_rgb_frame_number)}.jpg'
        #     cv2.imwrite(ui_output_image_file, final_result_to_show)
        #     cv2.waitKey(10)