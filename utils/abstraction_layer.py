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
import os


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
def display_intensity_maps(stack,config):
    mean_image = np.mean(stack, axis=2)  # Calculate the mean image

    mean_image = cv2.GaussianBlur(mean_image, (25, 25), 0)
    cv2.imwrite(config.output_folder + 'component_location_heatmap.png', assign_red_to_top_percentile(mean_image))
    # cv2.imshow('test', assign_red_to_top_percentile(mean_image))
    # cv2.waitKey(100)
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
    if not os.path.exists(config.output_folder):
        os.makedirs(config.output_folder)
    config.output_folder = config.output_folder + '/' + config.input_video.split('/')[-1].split('.')[0] + '/'
    if not os.path.exists(config.output_folder):
        os.makedirs(config.output_folder)
    else:
        print("WARNING: The output folder already exists. This means that the video has already been processed.")
    video_reader_object = video_reader(config)
    start_head_location = 0
    video_reader_object.skip_frames(start_head_location)
    # video_reader_object.total_number_of_rgb_frames = 530
    SIFT_processor = SIFT_bundle(config)
    Compo_DB = Component_Database()
    JSON_Processor = json_processor(config)
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
                                                                                   current_frame_grey, JSON_Processor, config)
        JSON_Processor.process_frame()
        # visualizer.visualize_components(current_frame_grey, detected_components, rgb=True, show=True, fill=False)

        # JSON_Processor.process_frame()
    video_reader_object.set_reader_head_to_frame_number(start_head_location)
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
        # visualizer.visualize_components(current_frame_grey, detected_components, rgb=True, show=True, fill=False)
        if component_fill_accumulator is None:
            component_fill_accumulator = np.array(component_image)
        if component_image is not None:
            component_fill_accumulator = np.dstack((component_fill_accumulator, component_image))


        if not os.path.exists(config.output_folder):
            os.makedirs(config.output_folder)

        JSON_Processor.write_json_to_file(Compo_DB)

        Save_plots_and_heatmpas(JSON_Processor, component_fill_accumulator, config)


def Save_plots_and_heatmpas(JSON_Processor, component_fill_accumulator, config):
    image_np = display_intensity_maps(np.array(component_fill_accumulator),config)
    # Visualize using cv2.imshow
    cv2.imwrite(config.output_folder + "intensity_map.jpg", image_np)
    # cv2.imshow("Intensity Maps", image_np)
    # cv2.waitKey(100)
    # sift = cv2.imread('sift.png')
    # cv2.imshow('sift', sift)
    # cv2.waitKey(100)
    # cv2.destroyAllWindows()

    import pandas as pd
    fd = JSON_Processor.get_stats()
    p = pd.DataFrame(fd[0], columns=['total_detected', 'area_filtered', 'overlap_filtered', 'sift_filtered'])
    plot = p.plot(title='compo detection stats')
    plot.set_xlabel("Frames x 10")
    plot.set_ylabel("Frequency")
    fig = plot.get_figure()
    fig.savefig(config.output_folder + "component_stats.png")
    plt.close()
    # sift = cv2.imread('s.png')
    # cv2.imshow('s', sift)
    # cv2.waitKey(100)

    p = pd.DataFrame(fd[1], columns=['total_detected', 'filtered'])
    plot = p.plot(title='database filter stats');
    # plot.title('SIFT Features across Frames')
    plot.set_xlabel("Frames x 10")
    plot.set_ylabel("Frequency")
    fig = plot.get_figure()
    fig.savefig(config.output_folder + "database_stats.png")
    plt.close()
    # sift = cv2.imread('d.png')
    # cv2.imshow('d', sift)
    # cv2.waitKey(100)

