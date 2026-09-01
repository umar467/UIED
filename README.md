# UIED (server-structured fork)

## Origin and licence

This repository is a derivative of [UIED](https://github.com/MulongXie/UIED) by
Mulong Xie and contributors, which is licensed under the Apache License, Version 2.0.

It has been restructured to run as a server-side pipeline over video input rather than
over single screenshots. The component detection (`detect_compo`), CNN classification
(`cnn`), configuration (`config`) and result processing (`result_processing`) are derived
from UIED, with modifications. `main.py`, `analyzer.py`, `color_utils.py` and `utils/`
are additions.

This repository is licensed under the Apache License, Version 2.0; see `LICENSE` for the
full text and `NOTICE` for attribution. As set out in that licence, the software is
provided on an "AS IS" basis, without warranties or conditions of any kind, and the
contributors accept no liability arising from its use.

---

## Code structured for web server

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