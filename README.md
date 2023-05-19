# Code Structured for Web Server

The file main.python abstracts away the functionality of the code. An exmaple use where a video file called 4.mp4 is copied into this folder would be 
'''
python main.py --input_video 4.mp4 --output_json_folder output.json
'''

Alternatively, both the parameters can be set directly in the config file as well.

The output structre is as follows:

json/video_name/
  - detections.json
  - sift.png
  - more graphs and plots

## Installing Software Dependencies

The conda_env.yml file in teh root should work out of the box, but if there is a problem the packages listed in the dependencies below can be manually installed as well.

Note, tensorflow wihtout gpu/cuda support falls back to the cpu and this is abstracted away from the code. So the code will still run regardless of GPU/CUDA availablity.

### Dependency
* **Python 3.9
* **NumPy
* **OpenCV**
* **PaddleOCR** (license check to be 100% confirmed)
* **Pandas**
* **Keras**
* **TensorFlow**

### Pre-trained CNN Weight Files

No longer required as using paddleocr now.
