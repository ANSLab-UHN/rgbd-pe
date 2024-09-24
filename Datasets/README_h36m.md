# H36M
Instructions specific to the Human3.6m dataset processing.

Need to be in mhp venv.

## 1. Download Human3.6M dataset

* Parsed data: https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE (RGB images and annotations)
* Depth data: http://vision.imar.ro/human3.6m/description.php
* Camera intrinsics: https://github.com/geopavlakos/c2f-vol-demo/issues/6


## 2. Clean up the downloaded data
Using **fix_h36m_gt.py**

This code takes original annotations file (as downloaded) in the annotations_ORIGINAL folder and removes the GT for the files that don't have depth.


## 3. Fix the h36m ground truth (MobileHumanPose version)
Using **create_mhp.py**

This code uses the downloaded annotations and creates the GT for the missing facial landmarks and converts everything to the COCO skeleton.


## 4. Create the Dite-HRNet equivalent ground truth
Using **create_dite_gt.py**

This code creates Dite-HRNet equivalent ground truth from the MobileHumanPose files.


## 5. Get rid of ground truth where the person is not visible in the frames
Using **out_of_bounds.py**


## 6. Correct ground truth
Using **create_mhp.py** and **create_dite_gt.py**


## 7. Create the RGB-D images
Using **get_depth.py**

This code created the rgbd png files but need **panutils.py** to be in the same folder to work!!!

## Extra code
* **plot_gt.py**: to check ground truth quality
* **gt_quality_check.py**: to see how much of the ground truth is out of bounds
* **draw_preds.py**: to check prediction quality