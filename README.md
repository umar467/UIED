## How to use?
In the setup folder I have added three .yml files. A first approach would be to attempt to recreate the evnironment from the env_cpu_full.yml and test the code.

To do so, make sure you have installed miniconda using either [with GPU Support](https://gretel.ai/blog/install-tensorflow-with-cuda-cdnn-and-gpu-support-in-4-easy-steps) or [cpu_only](https://educe-ubc.github.io/conda.html).

And then install the conda environment using the env_cpu_full.yml in the setup folder, as described in [this link](https://sachinjose31.medium.com/creating-an-environment-in-anaconda-through-a-yml-file-7e5deeb7676d).

In case this fails, you can try to manually install the dependencies. A single minor version number above or below the following package numbers should work fine.

### Dependency
* **Python 3.5**
* **Numpy 1.15.2**
* **Opencv 3.4.2**
* **Tensorflow 1.10.0**
* **Keras 2.2.4**
* **Sklearn 0.22.2**
* **Pandas 0.23.4**

### Weights and Video files

The weights and video files are shared at [this onedrive link](https://surreyac-my.sharepoint.com/:f:/g/personal/mf00963_surrey_ac_uk/EmbsgvakP2RCtANA0f7erQABp9t0AUuYV97Q-uYAdjwpFw?email=andy.woods%40rhul.ac.uk&e=LdrR0O). Let me know if the link doesn't work and we can fix that.

### Creating frames from the videos

Run ./extract_input_frames.sh to automatically generate the input frames.

### Test Run

Then try running run_single.py from inside the conda environment created above to test if everythign is working. You should see some images on screen ideally.


