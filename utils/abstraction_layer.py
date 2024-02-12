import timeit

import mkl

from utils.tqdm_callback import TqdmCallback

# mkl.set_num_threads(1)
import cv2
# cv2.setNumThreads(1)
import numpy as np
from scipy import ndimage
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
    video_reader_object = video_reader(config)
    start_head_location = video_reader_object.total_number_of_rgb_frames//2
    start_head_location = 500
    max_frames = 1100
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
    last_frame = video_reader_object.current_rgb_frame_number

    def progress(info: str):
        if config.progress_callback:
            config.progress_callback(info)
    pbar = tqdm(total=max_frames-start_head_location,  desc='Processing Video')

    analyzer = analyzer_class(config)
    pbar = TqdmCallback(total=video_reader_object.total_number_of_rgb_frames,
                        desc='Inspecting frames',
                        callback=config.progress_callback)

    while(video_reader_object.has_enough_frames()):

        current_frame_buffer_rgb, frame_numbers = video_reader_object.get_Frames()
        current_frame_buffer_grey = pre.conver_frames_to_grey(current_frame_buffer_rgb)
        current_frame_buffer_gradients = pre.conver_frames_to_gradient(current_frame_buffer_grey)
        common_gradients = pre.extract_common_gradients(current_frame_buffer_gradients)

        def get_connected_components(rgb, grad):
            # single images
            import copy
            rgb = copy.deepcopy(rgb)
            binary_image = pre.grad_to_binary(grad, 100)
            #output = cv2.connectedComponentsWithStats(binary_image, connectivity=4, ltype=cv2.CV_32S)
            # Although technically the same parameters are being passed above and below, the output is different,
            # perhaps becuase the litteral passing of parameters triggers different code execution paths.
            output = cv2.connectedComponentsWithStats(binary_image, 4, cv2.CV_32S)
            (numLabels, labels, stats, centroids) = output

            component_mask = np.zeros_like(grad)
            for i in range(1, numLabels): # first one is background
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]
                (cX, cY) = centroids[i]

                if area < 50:continue

                cv2.rectangle(component_mask, (x, y), (x + w, y + h), (1), -1)

            component_mask = np.stack([component_mask, component_mask, component_mask], axis=2)
            return component_mask


        def get_contour_highlights_for_detection_input():
            import copy
            rgb_t = copy.deepcopy(current_frame_buffer_rgb[0])
            # cv2.imshow('rgb', rgb_t)
            # cv2.waitKey(10)
            #
            blur = cv2.bilateralFilter(rgb_t, 9, 75, 75)

            # cv2.imshow('blurred_rgb', blur)
            # cv2.waitKey(10)


            # cv2.imshow('grey', current_frame_buffer_grey[0])
            # cv2.waitKey(10)

            # cv2.imshow('grad', current_frame_buffer_gradients[0])
            # cv2.waitKey(10)

            contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb_t, contours, -1, (0, 255, 0), 3)

            # cv2.imshow('cont_rgb_bin', rgb_t)
            # cv2.waitKey(10)

            grey_frame = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            gradn = pre.gray_to_gradient(grey_frame)
            binn = pre.grad_to_binary(gradn, 50)
            # binn = cv2.dilate(binn, None, iterations=5)

            contours, hierarchy = cv2.findContours(binn, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(blur, contours, -1, (0, 255, 0), thickness=3, hierarchy=hierarchy, maxLevel=1)

            # cv2.imshow('cont_blur_binn', blur)
            # cv2.waitKey(10)

            return rgb_t, blur


        def process_contours_for_detection_input():

            rgb_t, blur = get_contour_highlights_for_detection_input()

            grey_frame = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            gradn = pre.gray_to_gradient(grey_frame)
            binn = pre.grad_to_binary(grey_frame, config.minimum_gradient_difference)

            visualizer.show_frame(binary_image, use_cv=True, name='blur_inary_image')

            contours, hierarchy = cv2.findContours(binn, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for i in range(len(contours)):
                cv2.drawContours(blur, contours, -1, (0, 255, 0), thickness = 3, hierarchy = hierarchy, maxLevel = i)
                print(i)
                cv2.imshow('cont_blur_binn', blur)
                cv2.waitKey(10)

            det.component_detection_simplified_bfs(binary_image)
            det.component_detection_simplified_floodfill(binary_image)
            det.component_detection_simplified_floodfill_rgb(binary_image, current_frame_buffer_rgb[0])


        def get_std_mask(frames, percentile_filtering=None):
            # works for grey and gradient frames as well as rgb frame buffers
            std_rgb = np.std(frames, axis=0)
            if percentile_filtering is not None: # speed up this bit
                std_rgb = ndimage.percentile_filter(std_rgb, percentile=percentile_filtering, size=20)
            min_std_rgb = ndimage.minimum_filter(std_rgb, size=20)
            std_mask = min_std_rgb < 0.01
            std_filtered_frame = frames[0]*std_mask
            return std_mask, std_filtered_frame



        def get_ssim_mask(frames):

            before = frames[0].copy()
            after = frames[1].copy()

            # Convert images to grayscale
            before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
            after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)


            # Compute SSIM between two images
            from skimage.metrics import structural_similarity as ssim

            (score, diff) = ssim(before_gray, after_gray, full=True)
            print(diff.mean())
            diff_mask = diff < 0.8

            for i in range(2,len(frames),2):
                print(i)
                before = frames[i].copy()
                after = frames[i+1].copy()
                before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
                after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
                (score, diff) = ssim(before_gray, after_gray, full=True)
                print(diff.mean())
                diff[diff_mask] = False
                diff_mask = diff < 0.8
                print(diff.mean())


                print("Image similarity", score)
                diff = (diff * 255).astype("uint8")
                smask = diff <250
                diff[smask] = 0
                fmask = diff >250
                #cv2.imshow('diff', diff)
                #cv2.waitKey(100)

            fmask = np.stack([fmask, fmask, fmask], axis=2)
            visualizer.show_frame(current_frame_buffer_rgb[0] * fmask, use_cv=True, name='ssim_filtered')
            return fmask







        static_pixels, new_ui, sift_point_mask = SIFT_processor.get_static_pixels(current_frame_buffer_grey, JSON_Processor)


        binary_image = pre.convert_frame_to_binary(current_frame_buffer_gradients[0])
        binary_image = pre.grad_to_binary(current_frame_buffer_gradients[0], config.minimum_gradient_difference)
        visualizer.show_frame(binary_image, use_cv=True, name='binary_image')
        current_frame_rgb = current_frame_buffer_rgb[-1]
        current_frame_grey = current_frame_buffer_grey[-1]
        current_frame_number = frame_numbers[-1]
        frame_number = video_reader_object.current_rgb_frame_number
        text_components = Text_Extractor.detect_text_from_frame(current_frame_rgb)
        non_text_components = det.detect_components_from_binary_image(binary_image, static_pixels, JSON_Processor, detected_text_components=text_components, rgb_frame = current_frame_rgb)
        components = text_components + non_text_components
        detected_components = Compo_DB.compare_with_previously_detected_components(components, frame_number,
                                                                                   current_frame_grey, JSON_Processor, config,
                                                                                   force_check_previous_componenets=True)
        detection_frame = visualizer.visualize_components(current_frame_rgb, detected_components, rgb=True, show=True,
                                                          fill=False)
        analyzer.analyze_show(detected_components, current_frame_rgb,
                              video_reader_object.current_rgb_frame_number, Compo_DB.compos.copy(),
                              config, detection_frame, JSON_Processor, current_frame_number, video_reader_object)

        if new_ui:
            visualizer.new_ui_save(current_frame_buffer_rgb[0], video_reader_object.get_Frames()[-1], config)

        pbar.update(10)
        JSON_Processor.next_frame()
        visualizer.Save_plots_and_heatmpas(JSON_Processor, Compo_DB.compos.copy(), current_frame_grey, config)
        JSON_Processor.write_json_to_file(Compo_DB)




        if frame_number - last_frame > 100:
            last_frame = frame_number
            detection_frame = visualizer.visualize_components(current_frame_rgb, detected_components, rgb=True, show=False,
                                                              fill=False)
            analyzer.analyze_show(detected_components, current_frame_rgb,
                                                    video_reader_object.current_rgb_frame_number, Compo_DB.compos.copy(),
                                                    config, detection_frame, JSON_Processor, current_frame_number, video_reader_object)

    pbar.close()
