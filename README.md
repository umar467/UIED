# Code Structured for Web Server

The file main.python abstracts away the functionality of the code. An exmaple use where a video file called 4.mp4 is copied into this folder would be 
'''
python main.py --input_video 4.mp4 --output_json_path output.json
'''

## Installing Software Dependencies

The conda_env.yml file in teh root should work out of the box, but if there is a problem the packages listed in the dependencies below can be manually installed as well.

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
