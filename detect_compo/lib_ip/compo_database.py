import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config.CONFIG import Configuration
config = Configuration()

class Compo_Database:
    def __init__(self):
        self.config = config
        self.compos = []
        self.loaded_compos = 0
        self.statistics = []
        self.counter_id = 1
        self.ids_in_last_frame = []
        self.compo_change_cumulative = 0
        self.last_frame = None

    def compare_components(self, comp1, comp2, frame):
        relation = comp1.compo_relation(comp2)
        if relation != 0:
            component_present_in_frame = self.component_present_in_frame(comp2, frame)
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
    def compare_with_previously_detected_components(self, components, frame_number, frame, JSON_Processor, force_check_previous_componenets=True):
        if self.loaded_compos == 0:
            self.initialize_database(components, frame_number, frame)
            JSON_Processor.add_database_statistics_to_current_frame(self.compute_frame_statistics(components))
            return components
        updated_compos = []
        for component in components:
            for previous_component in self.compos:
                match = self.compare_components(component, previous_component, frame)
                if match:
                    previous_component.detected_in_frames.append(frame_number)
                    updated_compos.append(previous_component)
                    break
            if not match:
                component.id = self.counter_id
                self.counter_id+=1
                self.compos.append(component)
                self.loaded_compos+=1
                updated_compos.append(component)
        if force_check_previous_componenets:
            for previous_component in self.compos:
                if previous_component not in updated_compos:
                    if self.component_present_in_frame(previous_component, frame):
                        previous_component.detected_in_frames.append(frame_number)
                        updated_compos.append(previous_component)
                        #print('Addded extra 111')
        self.last_frame = frame
        JSON_Processor.add_database_statistics_to_current_frame(self.compute_frame_statistics(updated_compos))
        return updated_compos

    def component_present_in_frame(self, component, frame):
        # check if component crop is present in frame and last frame
        image_size = (128,128)
        bbox = component.bbox.put_bbox()
        crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        crop_last_frame = self.last_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        # resize crop and crop last frame to iamges_size
        crop = cv2.resize(crop, image_size)
        crop_last_frame = cv2.resize(crop_last_frame, image_size)
        from skimage.metrics import structural_similarity as ssim
        ssim = ssim(crop, crop_last_frame)
        # write a string to an image array of the same shape as crop
        # value = np.zeros_like(crop).astype(np.uint8)
        # value = cv2.putText(value, str(ssim), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # image_to_show = np.hstack([crop, crop_last_frame, value])
        # cv2.imshow('crop', image_to_show)
        # cv2.waitKey(1000)

        if ssim > config.ssim_threshold:
            return True
        return False
    def initialize_database(self, components, frame_number, frame):
        for component in components:
            component.detected_in_frames.append(frame_number)
            component.id = self.counter_id
            if component.category == 'Text':
                component.id = 'T_' + str(self.counter_id)
            self.counter_id += 1
        self.compos = components
        self.loaded_compos = len(self.compos)
        self.last_frame = frame
