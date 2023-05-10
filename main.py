#!/usr/bin/env python

import argparse
from config.CONFIG import Configuration
config = Configuration()
from utils.abstraction_layer import process_video
def parse_args():
    parser=argparse.ArgumentParser(description="a script to do stuff")
    parser.add_argument("--input_video", type=str, default='data/input/videos/1.mp4', required=False, help="File path for the input mp4 file. Best to give an absolute path like /usr/bin/../../file.mp4")
    parser.add_argument("--output_json_folder", type=str, default = 'json/', required=False, help="File path to store the output json file. Best to give an absolute path like /usr/bin/../../Detections.json")
    args=parser.parse_args()
    return args

def main():
    import detect_text_east.lib_east.eval as eval
    import detect_text_east.ocr_east as ocr
    models = eval.load()


    import cv2
    def resize_height_by_longest_edge(img_path, resize_length=800):
        org = cv2.imread(img_path)
        height, width = org.shape[:2]
        if height > width:
            return resize_length
        else:
            return int(resize_length * (height / width))

    resized_height = resize_height_by_longest_edge("image.jpg")

    ocr.east("image.jpg", '', models, 5, resize_by_height=resized_height, show=True)
    # inputs=parse_args()
    # print(f'Attempting to process video at "{inputs.input_video}" and output result to "{inputs.output_json_folder}"')
    # config.video_path=inputs.input_video
    # config.output_json_folder=inputs.output_json_folder
    # process_video(config)
    # print(f'Done processing video at "{inputs.input_video}" and outputs have been stored to "{inputs.output_json_path}"')

if __name__ == '__main__':
    main()