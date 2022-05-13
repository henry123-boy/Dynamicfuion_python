# Generating Masks using Salient Object Detection #

To limit the reconstruction output to the foreground of specific dynamic scenes, the background needs to be masked-out. Many scenes in datasets such as DeepDeform or VolumeDeform have backgrounds that can be easily separated from the foregrounds using a preset depth threshold. The thresholds can be manually set using the `far_clipping_distance` for each preset (see `data/presets.py`). However, to get a cleaner mask, we provide the option to pre-generate masks for each sequence using salience object detection (SOD).

Before generating the masks, make sure CMake has been run to configure the project (see above). Among other things, it will download the pretrained U^2 Net model to a local folder (~170 mb required).

To generate masks for a specific sequence within the DeepDeform dataset, run the U^2 Net salient object detector from the root of the repository like so:
```shell
 python3 run_sod.py -sp "train" -si 70 -o "sod" -d "/mnt/Data/Reconstruction/real_data/deepdeform"
```

Replace the split (-sp) & sequence index (-si) arguments above with those of the sequence you want to generate masks for, (-d) with the root directory of the DeepDeform dataset, and 'python3' with the name / path of your Python 3 executable. "-o" argument is the output folder and is "sod" by default.