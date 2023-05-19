#!/usr/bin/env python

import argparse
from config.CONFIG import Configuration
config = Configuration()
from utils.abstraction_layer import process_video
def parse_args():
    parser=argparse.ArgumentParser(description="a script to do stuff")
    parser.add_argument("--input_video", type=str, default='data/input/videos/0012.mov', required=False, help="File path for the input mp4 file. Best to give an absolute path like /usr/bin/../../file.mp4")
    parser.add_argument("--output_json_folder", type=str, default = 'json/', required=False, help="File path to store the output json file. Best to give an absolute path like /usr/bin/../../Detections.json")
    args=parser.parse_args()
    return args

def main():
    inputs=parse_args()
    print(f'Attempting to process video at "{inputs.input_video}" and output result to "{inputs.output_json_folder}"')
    config.input_video=inputs.input_video
    config.output_json_folder=inputs.output_json_folder
    process_video(config)
    print(f'Done processing video at "{inputs.input_video}" and outputs have been stored to "{inputs.output_json_folder}"')

if __name__ == '__main__':
    main()