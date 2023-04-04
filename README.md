### Source Code
This code is based on the publicly released UIED github repo from a specific commit from Jan 12, 2021 linked [here (GitHub Repo)](https://github.com/MulongXie/UIED/tree/814a8e70db69947ad5e3a25bbb411d7db8914cbc).

## Installing Software Dependencies
An Anaconda/miniconda setup is highly recommended on a linux machine with or withour a GPU/Nvidia-Drivers/Cuda. Although this can be done directly withot any virtual environments as well, or with just pip based virtual environments as well.

In the setup folder there are three .yml files which can be used to create an Anaconda environment with everything ready to go.

* env_full_cpu.yml. Assumes no GPU is present/ or the GPU should not be used and installs everything including tensorflow with CPU support only.
* env_full_gpu.yml. Install everything as above but Tensorflow has GPU support.
* env_simple_gpu.yml. This is a basic version of the above, in case any errors are encountered.

Note, tensorflow is only used to classify detected rectangles to a certain class of objects. If the classification flag is set to False, tensorflow is not required to begin with.

# Links which might be of some use:
* [Anaconda with GPU Support](https://gretel.ai/blog/install-tensorflow-with-cuda-cdnn-and-gpu-support-in-4-easy-steps) 
* [Anaconda with only cpu support](https://educe-ubc.github.io/conda.html).
* [Anaconda environment from .yml file](https://sachinjose31.medium.com/creating-an-environment-in-anaconda-through-a-yml-file-7e5deeb7676d).

In case this fails, you can try to manually install the dependencies. The version numbers don't matter exactly and the packages can be installed through either conda or pip from inside an Anaconda environment.

### Dependency
* **Python (Should come with the anaconda/pip environment)**
* **OpenCV**
* **Pandas**
* If classification using the CNN needs to be performed then install the following as well.
* **Keras**
* **TensorFlow**

### Pre-trained CNN Weight Files

The weights files are shared at [this onedrive link](https://surreyac-my.sharepoint.com/:f:/g/personal/mf00963_surrey_ac_uk/EmbsgvakP2RCtANA0f7erQABp9t0AUuYV97Q-uYAdjwpFw?email=andy.woods%40rhul.ac.uk&e=LdrR0O). Again, this is only needed if classification with CNN is being performed.

The downloaded folder 'data' can be placed in the base folder for this directory.

### Creating frames from the videos

First, place some .mp4 videos in the data/input/videos folder named numerically (1.mp4, 2.mp4 ...) and then run ./extract_input_frames.sh to automatically generate the input frames into the data/frames/ fodler.

### Test Run

Then try running run_pipeline.py from inside the conda environment created above to test if everythign is working. You should see some images on screen ideally.

## File structure Hints

*cnn/*
* Used to train classifier for graphic UI elements
* Set path of the CNN classification model

*config/*
* Set data paths 
* Set parameters for graphic elements detection

*data/*
* Input UI image frames and videos and output detection results
* Pre-Trained Model Weights

*detect_compo/*
* Graphic UI elemnts localization
* Graphic UI elemnts classification by CNN

*detect_text_east/*
* UI text detection by EAST

*result_processing/*
* Result evaluation and visualizition

*merge.py*
* Merge the results from the graphical UI elements detection and text detection 

*run_pipeline.py*
* Process images in the folder named 1 in data/input/frames.
