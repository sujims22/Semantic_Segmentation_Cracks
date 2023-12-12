# Semantic Segmentation of cracks
This project employs CNN and transformer models for semantic segmentation of building damage in Mexico City's 2017 earthquake, using annotated imagery to identify features like cracks and exposed rebar.


## Directory and File Descriptions

- `README.md`: Main documentation of the project
- `requirements.txt`: List of Python dependencies
- `.gitignore`: Git ignore file
- `Damage dataset/`: Images of structural damage and README
- `Data Processing/`: Data preprocessing scripts and notebooks
- `CNN Models/`: Convolutional neural network models that we used
- `Transformer Models/`: Transformer-based models that we used
- `Project_Presentation.pptx`: Contains presentation slides that offer a comprehensive overview of the project, including its aims, methods, and key results.




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

