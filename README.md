# Code Structured for Web Server

The file main.python abstracts away the functionality of the code. An exmaple use where a video file called 4.mp4 is copied into this folder would be 
'''
python main.py --input_video 4.mp4 --output_json_path output.json
'''

## Installing Software Dependencies

In the setup folder there are three additional .yml files which can be used to create an Anaconda environment with everything ready to go.

* env_full_cpu.yml. Assumes no GPU is present/ or the GPU should not be used and installs everything including tensorflow with CPU support only.
* env_full_gpu.yml. Install everything as above but Tensorflow has GPU support.
* env_simple_gpu.yml. This is a basic version of the above, in case any errors are encountered.

Note, tensorflow wihtout gpu/cuda support falls back to the cpu and this is abstracted away from the code. So the code will still run regardless of GPU/CUDA availablity.

### Dependency
* **Python 3.9
* **NumPy
* **OpenCV**
* **Pandas**
* **Keras**
* **TensorFlow**

### Pre-trained CNN Weight Files

The weights files are shared at [this onedrive link](https://surreyac-my.sharepoint.com/:f:/g/personal/mf00963_surrey_ac_uk/EmbsgvakP2RCtANA0f7erQABp9t0AUuYV97Q-uYAdjwpFw?email=andy.woods%40rhul.ac.uk&e=LdrR0O).

The downloaded folder 'data' can be placed in the base folder for this directory.
