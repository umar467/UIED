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

    def compare_components(self, comp1, comp2):
        relation = comp1.compo_relation(comp2)
        if relation != 0:
            return True
        return False

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