# Semantic Segmentation of cracks
This project involves analyzing images of buildings damaged in the Mexico City 2017 Earthquake. The dataset `imageDamage.zip` contains pixel-level annotations of various types of damage.

## Data Sources
Images are gathered from:
1. [DataCenterHub](https://datacenterhub.org/resources/14746)
2. Photographed by Vedhus Hoskere

## Annotations
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


The statistics of the number of pixels annotated with each of the classes is provided in the Statistics folder
There is also an incomplete inspection file which has per-image inspection data for some of the images.

Another set of classes for the images, but this project is not of this focus
| Class                | Color          |
|----------------------|----------------|
| Other                | black          |
| Sky                  | deepskyblue    |
| Building             | khaki          |
| Sidewalk             | gainsboro      |
| Road                 | darkgrey       |
| Sign/Pole/Light      | yellow         |
| Tree                 | springgreen    |
| Vehicle              | purple         |
| Wall                 | khaki          |
| Foundation           | gainsboro      |
| Window               | azure          |
| Door                 | sienna         |
| Column               | royalblue      |
| Beam                 | seagreen       |
| Balcony              | purple         |
| Void                 | orangered      |


###  Use PIL to open images with single integer at each pixel as opposed to a color image ###
###  The index can be used to identify the corresponding class type 

```python

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

im = Image.open(os.path.join(folder,file))
image = np.array(im)
plt.subplot(image)
