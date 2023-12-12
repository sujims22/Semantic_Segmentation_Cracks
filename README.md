# Semantic Segmentation of cracks
This project employs CNN and transformer models for semantic segmentation of building damage in Mexico City's 2017 earthquake, using annotated imagery to identify features like cracks and exposed rebar.


## Directory and File Descriptions

- `src/`: Contains all the source code for the project. The `main.py` file is the entry point of the application, and `module.py` is a sample module demonstrating the code structure.
- `data/`: This directory is intended for storing project data. Note: Due to size and privacy concerns, actual data files are not included in the repository.
- `docs/`: Holds documentation for the project, including detailed API descriptions.
- `tests/`: Contains test scripts and files, ensuring the codebase remains stable and functional as changes are made.
- `notebooks/`: Jupyter notebooks for demonstrating usage examples and for conducting exploratory data analysis.
- `requirements.txt`: Lists all the Python dependencies required to run the project.
- `.gitignore`: Prevents specific files and directories from being tracked by Git (e.g., confidential data, system files).
- `LICENSE`: The license file specifying the terms under which the project can be used.
- `README.md`: Provides an overview of the project, setup instructions, and other essential information.

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

