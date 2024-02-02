import mkl

from utils.tqdm_callback import TqdmCallback

mkl.set_num_threads(1)
import cv2
cv2.setNumThreads(1)
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

        # frames = video_reader_object.get_all_Frames()
        # gframes = pre.conver_frames_to_grey(frames)
        # fstd = gframes.std(axis=0)
        # #visualizer.show_frame(fstd)
        #
        # from skimage.metrics import structural_similarity as ssim
        # import numpy as np
        #
        # for i in range(0, len(gframes)-1, 10):
        #     #fstd = gframes[i:i+50].std(axis=0)
        #     fstd = np.clip(fstd, 0, 255).astype(int)
        #     #fstd = fstd*(fstd<10)
        #     ssim_score, ssim_image = ssim(gframes[i], fstd, full=True)
        #     ssim_score_2, ssim_image_2 = ssim(gframes[i], gframes[i+1], full=True)
        #     ssim_image = (fstd<50)*ssim_image_2
        #     # pp = gframes[i]*(ssim_image>0.85)
        #     # pp = np.clip(pp,0,255)
        #     # pp = pp*(pp>100)
        #
        #     visualizer.show_frame(fstd.astype(float), use_cv=True, name='fstd')
        #     visualizer.show_frame(gframes[i], use_cv=True, name='frame')
        #     visualizer.show_frame(ssim_image, use_cv=True, name='ssim')
        #     visualizer.show_frame(ssim_image_2, use_cv=True, name='ssim2')
        #
        # # ssim_score, ssim_image = ssim(gframes[131], gframes[554], full=True)
        # # visualizer.show_frame(np.hstack([gframes[131], gframes[554],ssim_image]))

        current_frame_buffer_rgb = video_reader_object.get_Frames()
        current_frame_buffer_grey = pre.conver_frames_to_grey(current_frame_buffer_rgb)
        current_frame_buffer_gradients = pre.conver_frames_to_gradient(current_frame_buffer_grey)
        common_gradients = pre.extract_common_gradients(current_frame_buffer_gradients)
        #cistpm operations


        min_std_common_gradients = ndimage.minimum_filter(common_gradients, size=2)
        # min_std_common_gradients = cv2.dilate(min_std_common_gradients, np.ones((3,3)))
        # visualizer.show_frame(min_std_common_gradients, use_cv=True, name='min_std_common_gradients')

        percentile_std_common_gradients = ndimage.percentile_filter(common_gradients, percentile=20, size=2)
        # percentile_std_common_gradients = cv2.dilate(percentile_std_common_gradients, np.ones((3,3)))
        # visualizer.show_frame(percentile_std_common_gradients, use_cv=True, name='percentile_std_common_gradients')

        std_rgb = np.std(current_frame_buffer_rgb, axis=0)
        std_grey = np.std(current_frame_buffer_grey, axis=0)
        std_grad = np.std(current_frame_buffer_gradients, axis=0)

        min_std_rgb = ndimage.minimum_filter(std_rgb, size=20)
        min_std_grey = ndimage.minimum_filter(std_grey, size=20)
        min_std_grad = ndimage.minimum_filter(std_grad, size=20)

        mask = min_std_rgb < 0.01
        showy = current_frame_buffer_rgb[0]*mask


        before = current_frame_buffer_rgb[0].copy()
        after = current_frame_buffer_rgb[1].copy()

        # Convert images to grayscale
        before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)


        # Compute SSIM between two images
        from skimage.metrics import structural_similarity as ssim

        (score, diff) = ssim(before_gray, after_gray, full=True)
        print(diff.mean())
        diff_mask = diff < 0.8

        for i in range(2,10,2):
            print(i)
            before = current_frame_buffer_rgb[i].copy()
            after = current_frame_buffer_rgb[i+1].copy()
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
            cv2.waitKey(100)
        fmask = np.stack([fmask,fmask,fmask],axis=2)
        visualizer.show_frame(current_frame_buffer_rgb[0] * fmask, use_cv=True, name='ssim_filtered')
        visualizer.show_frame(showy, use_cv=True, name='std_rgb')
        # percentile_std_rgb = ndimage.percentile_filter(std_rgb, percentile=80, size=20)
        # percentile_std_grey = ndimage.percentile_filter(std_grey, percentile=80, size=20)
        # percentile_std_grad = ndimage.percentile_filter(std_grad, percentile=80, size=20)
        # visualizer.show_frame(percentile_std_rgb, use_cv=True, name='percentile_std_rgb')
        # visualizer.show_frame(percentile_std_grey, use_cv=True, name='percentile_std_grey')
        # visualizer.show_frame(percentile_std_grad, use_cv=True, name='percentile_std_grad')
        #
        # visualizer.show_frame(min_std_rgb, use_cv=True, name='min_std_rgb')
        # visualizer.show_frame(min_std_grey, use_cv=True, name='min_std_grey')
        # visualizer.show_frame(min_std_grad, use_cv=True, name='min_std_grad')



        static_pixels, new_ui = SIFT_processor.get_static_pixels(current_frame_buffer_grey, JSON_Processor)
        binary_image = pre.convert_frame_to_binary(current_frame_buffer_gradients[0])
        binary_image = pre.grad_to_binary(current_frame_buffer_gradients[0], config.minimum_gradient_difference)
        #visualizer.show_frame(binary_image, use_cv=True, name='binary_image')
        current_frame_rgb = current_frame_buffer_rgb[-1]
        current_frame_grey = current_frame_buffer_grey[-1]
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
                              config, detection_frame, JSON_Processor)
        #if new_ui:
        #     visualizer.new_ui_save(current_frame_buffer_rgb[0], video_reader_object.get_Frames()[-1], config)
        #
        # pbar.update(10)
        # JSON_Processor.next_frame()
        # visualizer.Save_plots_and_heatmpas(JSON_Processor, Compo_DB.compos.copy(), current_frame_grey, config)
        # JSON_Processor.write_json_to_file(Compo_DB)




        # if frame_number - last_frame > 100:
        #     last_frame = frame_number
        #     detection_frame = visualizer.visualize_components(current_frame_rgb, detected_components, rgb=True, show=False,
        #                                                       fill=False)
        #     analyzer.analyze_show(detected_components, current_frame_rgb,
        #                                             video_reader_object.current_rgb_frame_number, Compo_DB.compos.copy(),
        #                                             config, detection_frame, JSON_Processor)

    pbar.close()
