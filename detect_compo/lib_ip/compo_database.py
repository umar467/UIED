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

    def compare_components(self, comp1, comp2):
        relation = comp1.compo_relation(comp2)
        if relation != 0:
            return True
        return False

    def compute_frame_statistics(self, components, frame_number):
        if len(self.ids_in_last_frame)==0:
            for compo in components:
                self.ids_in_last_frame.append(compo.id)
            return False
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

        return new_ui
    def compare_with_previously_detected_components(self, components, frame_number):
        if self.loaded_compos == 0:
            for component in components:
                component.detected_in_frames.append(frame_number)
                component.id = self.counter_id
                if component.category == 'Text':
                    component.id = 'T_' + str(self.counter_id)
                self.counter_id+=1
            self.compos = components
            self.loaded_compos = len(self.compos)
            return components
        updated_compos = []
        for component in components:
            for previous_component in self.compos:
                match = self.compare_components(component, previous_component)
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
        return updated_compos