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
| Filled Cracks        | lightgrey  |
| Shear Cracks         | red        |
| Plaster Cracks       | pink       |
| OCracks              | cyan       |
| Concrete Exposed Rebar | orange   |
| Cracks               | red        |
| Exposed Rebar        | orange     |

This table combines both FD and FDR categories along with their corresponding colors for a unified representation.
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

This table provides the names of different concrete damage classes along with their corresponding colors.

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
