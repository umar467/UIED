import mkl
# mkl.set_num_threads(1)
import cv2
# cv2.setNumThreads(1)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config.CONFIG import Configuration
config = Configuration()
import detect_compo.lib_ip.ip_preprocessing as pre


class Compo_Database:
    def __init__(self):
        self.config = config
        self.compos = []
        self.loaded_compos = 0
        self.processed_frames = 0
        self.statistics = []
        self.counter_id = 1
        self.ids_in_last_frame = []
        self.compo_change_cumulative = 0
        self.last_frame = None


    def compare_components(self, comp1, comp2, frame):
        if comp1.category != comp2.category:
            return False
        if comp1.category == 'text':
            if comp1.content != comp2.content:
                return False
        relation = comp1.compo_relation(comp2)
        if relation != 0:
            component_present_in_frame = self.component_present_in_frame(comp2, frame, comp2.bbox.put_bbox())
            if component_present_in_frame:
                return True
            if not component_present_in_frame:
                return False
        return False

    def compute_frame_statistics(self, components):
        if len(self.ids_in_last_frame)==0:
            for compo in components:
                self.ids_in_last_frame.append(compo.id)
            return [0, 0, []]
        total_matched = 0
        total_new = 0
        ids_in_current_frame = []
        for component in components:
            ids_in_current_frame.append(component.id)
            if component.id in self.ids_in_last_frame:
                total_matched+=1
            else:
                total_new+=1
        if total_matched!=0:
            new_compo_ratio = total_new / total_matched
        else:
            new_compo_ratio = 0
        # print(new_compo_ratio)
        self.compo_change_cumulative+=new_compo_ratio
        self.ids_in_last_frame.append(ids_in_current_frame)
        new_ui = False
        if self.compo_change_cumulative > config.new_UI_layout_change_ratio:
            new_ui = True
            self.compo_change_cumulative = 0

        return [total_matched, total_new, ids_in_current_frame]
    def compare_with_previously_detected_components(self, components, frame_number, frame, JSON_Processor, config,  force_check_previous_componenets=True):
        force_check_added = 0
        duplicate_removed = 0
        if self.loaded_compos == 0:
            self.initialize_database(components, frame_number, frame, config)
            JSON_Processor.add_database_statistics_to_current_frame(self.compute_frame_statistics(components))
            return components
        updated_compos = []
        for component in components:
            for previous_component in self.compos:
                match = self.compare_components(component, previous_component, frame)
                if match:
                    previous_component.detected_in_frames.append(frame_number)
                    previous_component.bbox_historical.append(component.bbox.put_bbox())
                    updated_compos.append(previous_component)
                    break
            if not match:

                if component.category == 'Text':
                    component.id = 'T_' + str(self.counter_id)
                else:
                    component.id = self.counter_id
                self.counter_id+=1
                self.compos.append(component)
                self.loaded_compos+=1
                updated_compos.append(component)
                self.save_component_as_png(component, frame, config)
        if force_check_previous_componenets:
            for previous_component in self.compos:
                if previous_component not in updated_compos:
                    if previous_component.width/previous_component.height < 2:
                        if self.component_present_in_frame_historic(previous_component, frame):
                            previous_component.detected_in_frames.append(frame_number)
                            previous_component.bbox_historical.append(previous_component.bbox.put_bbox())
                            updated_compos.append(previous_component)
                            #print('Addded extra 111')
                            force_check_added+=1

        # check duplicate components
        for component in updated_compos:
            for previous_component in updated_compos:
                if component != previous_component:
                    match = self.compare_components(component, previous_component, frame)
                    if match:
                        for frame_no_p in previous_component.detected_in_frames:
                            component.detected_in_frames.append(frame_no_p)
                        for bbox_historical in previous_component.bbox_historical:
                            component.bbox_historical.append(bbox_historical)
                        updated_compos.remove(previous_component)
                        #print('Removed duplicate')
                        duplicate_removed+=1

        # Remove invalid components
        for component in updated_compos:
            bbox = component.bbox.put_bbox()
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                updated_compos.remove(component)

        self.last_frame = frame
        db_stats_current = self.compute_frame_statistics(updated_compos)
        JSON_Processor.add_database_statistics_to_current_frame(db_stats_current)
        self.processed_frames += 1
        print(f'Frame {frame_number} - {len(updated_compos)} components detected, {force_check_added} force check added, {duplicate_removed} duplicates removed')
        return updated_compos

    def component_present_in_frame_historic(self, component, frame):
        return self.component_present_in_frame(component, frame, component.bbox.put_bbox())
        if len(component.detected_in_frames) > 5:
            look_up_bboxes = component.bbox_historical[-5:]
        else:
            look_up_bboxes = component.bbox_historical
        for bbox in look_up_bboxes:
            if self.component_present_in_frame(component, frame, bbox):
                return True
        return False
    def component_present_in_frame(self, component, frame, bbox):
        # check if component crop is present in frame and last frame
        image_size = (128,128)
        # bbox = component.bbox.put_bbox()
        crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        crop_last_frame = self.last_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        # resize crop and crop last frame to iamges_size
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            return False
        crop = cv2.resize(crop, image_size)
        crop_last_frame = cv2.resize(crop_last_frame, image_size)

        crop = pre.gray_to_gradient(crop)
        crop_last_frame = pre.gray_to_gradient(crop_last_frame)

        from skimage.metrics import structural_similarity as ssimer
        ssim = ssimer(crop, crop_last_frame)
        # write a string to an image array of the same shape as crop

        value = np.zeros_like(crop).astype(np.uint8)
        value = cv2.putText(value, str(ssim), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        image_to_show = np.hstack([crop, crop_last_frame, value])

        def contrast_stretch(image):
            # Calculate the minimum and maximum pixel values
            min_val = np.min(image)
            max_val = np.max(image)

            # Perform contrast stretching
            stretched_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

            return stretched_image
        crop = contrast_stretch(crop)
        crop_last_frame = contrast_stretch(crop_last_frame)
        ssim_n = ssimer(crop, crop_last_frame)

        value = np.zeros_like(crop).astype(np.uint8)
        value = cv2.putText(value, str(ssim_n), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        image_to_show = np.hstack([ image_to_show, crop, crop_last_frame, value])
        # cv2.imshow('crop', image_to_show)
        # cv2.waitKey(1000)

        ssim = ssim_n
        if ssim > config.ssim_threshold:
            return True
        return False
    def initialize_database(self, components, frame_number, frame, config):
        for component in components:
            component.detected_in_frames.append(frame_number)
            component.bbox_historical.append(component.bbox.put_bbox())
            component.id = self.counter_id
            if component.category == 'Text':
                component.id = 'T_' + str(self.counter_id)
            self.counter_id += 1
            self.save_component_as_png(component, frame, config)
        self.compos = components
        self.loaded_compos = len(self.compos)
        self.last_frame = frame
    def get_all_components(self):
        return self.compos

    def save_component_as_png(self, component, frame, config):
        bbox = component.bbox.put_bbox()
        crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        # reshape crop to 128x128 while preserving the aspect ratio
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            return
        crop = pre.resize_by_height(crop, config.component_png_size[0])
        # crop = cv2.resize(crop, config.component_png_size)
        output = config.output_folder + '/component_crops/'
        import os
        if not os.path.exists(output):
            os.makedirs(output)
        cv2.imwrite(output +  str(component.id)+'.png', crop)