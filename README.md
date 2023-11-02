# Deep_Learning_Project

The zip file imageDamage.zip contains pixel-level annotations of damage of buildings damaged in the Mexico City 2017 Earthquake

Images have been gethered from two sources:
1. https://datacenterhub.org/resources/14746 and 
2. Photographed by Vedhus Hoskere

The annotations are in PNG format. Two sets of annotaitons are availble for each image
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


scene_names = ['Other', 'Sky', 'Building', 'Sidewalk', 'Road', 'Sign/Pole/Light', 'Tree', 'Vehicle']
scene_colors = ['black','deepskyblue','khaki','gainsboro','darkgrey','yellow','springgreen','purple']
component_names = ['Other', 'Sky','Wall', 'Foundation', 'Window', 'Door', 'Column', 'Beam', 'Balcony', 'Void']
component_colors = ['black','black','khaki','gainsboro','azure','sienna','royalblue','seagreen','purple','orangered']


FDR_names =  ['No', 'Scratches', 'Grooves/Joints', 'Cables', 'Filled Cracks', 'Cracks', 'Exposed Rebar']
FDR_colors = ['black', 'red','green','white','lightgrey','red','orange']
CD_names =  ['No', 'Shadows', 'Dirt', 'Vegetative Growth', 'Debris', 'Marks', 'Spalling', 'Voids']        
CD_colors = ['black', 'grey','goldenrod','springgreen','fuchsia','purple','tomato','yellow']



Not sure if the color names are accurate. 

###  Use PIL to open images with single integer at each pixel as opposed to a color image ###
###  The index can be used to identify the corresponding class type #########################



import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

im = Image.open(os.path.join(folder,file))
image = np.array(im)
plt.subplot(image)