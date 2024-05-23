# Code Structured for Web Server

## Running the Code

The code can be run via the command line as follows:

```
python main.py --input_video /scartch/videos/4.mp4 --output_json_folder /scratch/results/
```

It's best to use absolute path for the input and output folders. 

If the output folder structure adds subfolders for the video input path just change the linux or windows output path option in main.py line 21-24. 

## Installing Software Dependencies

Note, this will run regardless of GPU/CUDA availablity.

The dependencies in the requirements.txt file should be installed via pip and python 3.9. 

If for some reason they do not work for you then there are three .yml files in the reproduce_software_environment folder which list explicit package numbers to reproduce our software environment on linux or windows machines.

## NOTE

This code was last tested on windows 11 build 22631.3593 on 23/05/2024 should that be useful for debugging purposes.