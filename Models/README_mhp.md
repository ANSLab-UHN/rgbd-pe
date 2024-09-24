# MobileHumanPose Model
Instructions to set up a venv for this model, train and test it, and backup for the configuration and loading files.

Need to be in mhp venv.

Note: LPSKI was used as the RGB baseline for this model


## 1. Clone Github repository
https://github.com/SangbumChoi/MobileHumanPose


## 2. Prepare venv
Use the **mhp_requirements.txt** file to create the venv with the packages necessary for this model.


## 3. Add the backbone files
Add the backbone files into common/backbone.

Update the __init__.py file accordingly.


## 4. Add the dataset files
Add the data folders into data.

Change the file and annotation paths as needed


## 5. File replacement
Replace the **model.py** file (in main/)

Replace the **dataset.py** file (in dataset/)

## Good to go now!!!
