import cv2
import numpy as np
import os
import json
import detect_compo.lib_ip.ip_preprocessing as pre
from detect_compo.lib_ip.video_utils import video_reader
import detect_compo.lib_ip.ip_detection as det
import detect_compo.lib_ip.visualize_util as visualizer


def process_video(config):
    outputs = []
    video_reader_object = video_reader(config)
    frame = video_reader_object.get_processed_frame()

    count = 0
    logg = pre.gray_to_gradient(frame[2])
    while frame is not None:
        frame = video_reader_object.get_processed_frame()
        if frame is None:
            break

        raw = frame[2]
        raw = pre.gray_to_gradient(raw)
        ogg = raw

        div = 64
        ans = ogg & logg
        ogg = ans

        logg = ogg
        count +=1

        if count>20:
            count=0
            logg = raw

            ogg2 = pre.grad_to_binary(ogg, min=20)
            ogg2 = cv2.dilate(ogg2, None, iterations=2)

            components = det.component_detection(ogg2, min_obj_area=config.min_object_area)
            ogg3 = visualizer.visualize_components(frame, components, show=False, rgb=False)
            json_current_Frame = visualizer.get_json(frame, components)
            if json_current_Frame is not None:
                outputs.append(json_current_Frame)


    name = 'Video ' + str(config.video_path)
    output = {name: outputs}
    if os.path.exists(config.output_json_path):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    f_out = open(config.output_json_path, append_write)

    json.dump(output, f_out, indent=4)