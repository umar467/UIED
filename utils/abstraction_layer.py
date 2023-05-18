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


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()

def assign_red_to_top_percentile(image):
    # Calculate the threshold value for the top 20% percentile
    percentile_threshold = np.percentile(image, 95)

    # Convert grayscale image to color image
    color_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Assign red color (0, 0, 255) to the pixels above the threshold
    color_image[np.where(image > percentile_threshold)] = [0, 0, 255]

    return color_image
def display_intensity_maps(stack):
    mean_image = np.mean(stack, axis=2)  # Calculate the mean image

    mean_image = cv2.GaussianBlur(mean_image, (25, 25), 0)
    cv2.imshow('test', assign_red_to_top_percentile(mean_image))
    cv2.waitKey(100)
    # Create heatmap using seaborn
    sns.set()
    fig, ax = plt.subplots(figsize=(8, 8))
    heatmap = sns.heatmap(mean_image, cmap='hot', ax=ax)
    heatmap.set_title('Pixel Heatmap')
    heatmap.set_xlabel('Columns')
    heatmap.set_ylabel('Rows')

    # Save the figure as a NumPy array
    fig.canvas.draw()
    img_np = np.array(fig.canvas.renderer.buffer_rgba())

    plt.close()

    return img_np



def process_video(config):
    video_reader_object = video_reader(config)
    video_reader_object.skip_frames(500)
    video_reader_object.total_number_of_rgb_frames = 600
    SIFT_processor = SIFT_bundle(config)
    Compo_DB = Component_Database()
    JSON_Processor = json_processor()
    Text_Extractor = Text_Processor(config)
    component_fill_accumulator = None
    while(video_reader_object.has_enough_frames()):
        print(video_reader_object.current_rgb_frame_number)
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
                                                                                   current_frame_grey, JSON_Processor)
        visualizer.visualize_components(current_frame_grey, detected_components, rgb=True, show=True, fill=False)

    video_reader_object.set_reader_head_to_frame_number(500)
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
                                                                                   JSON_Processor)

        component_image = visualizer.visualize_components(current_frame_grey, detected_components, rgb=False, show=True, fill=True)
        visualizer.visualize_components(current_frame_grey, detected_components, rgb=True, show=True, fill=False)
        if component_fill_accumulator is None:
            component_fill_accumulator = np.array(component_image)
        if component_image is not None:
            # stack = np.dstack((stack, component_image))
            # component_fill_accumulator.append(component_image)
            component_fill_accumulator = np.dstack((component_fill_accumulator, component_image))


        # JSON_Processor.produce_json_for_frame_detections(detected_components, frame_number, config)
        #
        # JSON_Processor.write_json_to_file(Compo_DB)

        image_np = display_intensity_maps(np.array(component_fill_accumulator))
        # image_np = display_intensity_maps(image_stack)

        # Visualize using cv2.imshow
        cv2.imshow("Intensity Maps", image_np)
        cv2.waitKey(100)
        # cv2.destroyAllWindows()