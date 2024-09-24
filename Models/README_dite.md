# Dite-HRNet Model
Instructions to set up a venv for this model, train and test it, and backup for the configuration and loading files.

Need to be in dite venv.


## 1. Clone Github repository
https://github.com/ZiyiZhang27/Dite-HRNet


## 2. Prepare venv
Use the **dite_requirements.txt** file to create the venv with the packages necessary for this model.


## 3. Modify mmpose files in the venv
Files need to be added in mmpose venv library files.


### **BACKBONES**
Copy the backbone files into venv/_dite/lib/python3.8/site-packages/mmpose/models/backbones
* gloditehrnet - correspond to CH
* dite_ezcat -corresponds to Cat
* dite_fusev2 -corresponds to Fuse

Update the __init__.py files in the folder to include:

`from .gloditehrnet import GLODiteHRNet`

`from .dite_ezcat import ezCAT_DiteHRNet`

`from .dite_fusev2 import Fusev2_DiteHRNet`

Add the model names in the all function in the same __init__.py file


### **IMAGE LOADING**
Copy the image loading file into venv/_dite/lib/python3.8/site-packages/mmpose/datasets/pipelines. LoadRGBD was created to allow the models to load the RGB-D files.

Update the __init__.py files in the folder to include:

`from .loadingRGBD import LoadRGBD`

## Good to go now!!!
