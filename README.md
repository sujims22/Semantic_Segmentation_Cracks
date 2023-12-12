# Semantic Segmentation of cracks
This project employs CNN and transformer models for semantic segmentation of building damage in Mexico City's 2017 earthquake, using annotated imagery to identify features like cracks and exposed rebar.


## Directory and File Descriptions

- `CNN Models/`: This directory contains the convolutional neural network models developed for the project, featuring different architectures for semantic segmentation.
- `Damage dataset/`: Includes the dataset with images of structural damage. The README within this folder provides details on the dataset's characteristics and usage guidelines.
- `Data Processing/`: Houses scripts and Jupyter notebooks used for preprocessing the data, preparing it for analysis and model training.
- `requirements.txt`: A crucial file listing all the Python dependencies needed to run the project. This file should be used to set up an appropriate Python environment.
- `.gitignore`: Tells Git which files or directories to ignore, preventing unnecessary files from being included in the repository.
- `Project_Presentation.pptx`: Contains presentation slides that offer a comprehensive overview of the project, including its aims, methods, and key results.
- `README.md`: The main document that provides a detailed description of the project, instructions for setting up and running it, and other essential information for users and contributors.



## Data Sources
Images are gathered from:
1. [DataCenterHub](https://datacenterhub.org/resources/14746)
2. Photographed by Vedhus Hoskere

### Annotations
Two sets of annotations are available in PNG format for each image:

1. Fine damage and damage-like features, including features like cracks, exposed rebar, cables, etc
            Class names: 'No', 'Scratches', 'Grooves/Joints', 'Cables', 'Filled Cracks', 'Cracks', 'Exposed Rebar'

| Class                | Color      |
|----------------------|------------|
| No                   | black      |
| Scratches            | red        |
| Grooves/Joints       | green      |
| Cables               | white      |
| Filled Cracks        | lightgrey  |  |
| Cracks               | red        |
| Exposed Rebar        | orange     |


2. Coarse damage and damage-like features, including spalling, dirt, etc.
            Class names: 'No', 'Shadows', 'Dirt', 'Vegetative Growth', 'Debris', 'Marks', 'Spalling', 'Voids'

| Class                | Color      |
|----------------------|------------|
| No                   | black      |
| Shadows              | grey       |
| Dirt                 | goldenrod  |
| Vegetative Growth    | springgreen|
| Debris               | fuchsia    |
| Marks                | purple     |
| Spalling             | tomato     |
| Voids                | yellow     |

###  Use PIL to open images with single integer at each pixel as opposed to a color image ###
###  The index can be used to identify the corresponding class type 

```python

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

im = Image.open(os.path.join(folder,file))
image = np.array(im)
plt.subplot(image)

