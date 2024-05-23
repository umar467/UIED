#!/usr/bin/env python

import argparse
from config.CONFIG import Configuration
config = Configuration()
from utils.abstraction_layer import process_video
import os
def parse_args():
    parser=argparse.ArgumentParser(description="a script to do stuff")
    parser.add_argument("--input_video", type=str, default='data/input/videos/11.mp4', required=False, help="File path for the input mp4 file. Best to give an absolute path like /usr/bin/../../file.mp4")
    parser.add_argument("--output_folder", type=str, default = 'processed_results/', required=False, help="File path to store the output json file. Best to give an absolute path like /usr/bin/../../Detections.json")
    parser.add_argument("--show_debug", type=bool, default=False, required=False, help="Show debug visual during program execution (default: False)")
    args=parser.parse_args()
    return args

def main():
    inputs=parse_args()
    print(f'Attempting to process video at "{inputs.input_video}" and output result to "{inputs.output_folder}"')
    config.input_video=inputs.input_video
    config.output_folder=inputs.output_folder
    config.show_debug=inputs.show_debug

    # Linux output path
    # config.output_folder = config.output_folder + os.sep + config.input_video.split(os.sep)[-1].split('.')[0] + os.sep
    # Windows Output Path
    config.output_folder = config.output_folder + os.sep + \
                           config.input_video.split(os.sep)[-1].split('/')[-1].split('.')[0] + os.sep

    process_video(config)
    print(f'Done processing video at "{inputs.input_video}" and outputs have been stored to "{inputs.output_folder}"')

if __name__ == '__main__':
    main()
