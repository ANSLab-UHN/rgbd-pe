import os
import cv2
import json
import panutils
import numpy as np
from tqdm import tqdm

og_p = '/media/veracrypt2/gloria/H36M/annotations_USEME/mhp/'
new_p = '/media/veracrypt2/gloria/H36M/annotations_USEME/dite/'

# Also added test subset!!
t = input('Enter data type: [1] for Train, [2] for Val, [3] for Test:         ')
if str(t) == '1':
    subjects = ['1', '5', '6', '7', '8']
    new_path = new_p + 'dite_h36m_train_gt.json'
elif str(t) == '2':
    subjects = ['9']
    new_path = new_p + 'dite_h36m_val_gt.json'
elif str(t) == '3':
    subjects = ['11']
    new_path = new_p + 'dite_h36m_test_gt.json'
else:
    subjects = '0'
    print('ERROR: unrecognized data type')

coco_dict = {"images": [], "annotations": [], "categories": []}

for subject in subjects:
    print('*' * 150)
    print('LOOKING AT SUBJECT:          ', subject)
    cp = og_p + 'Human36M_subject' + subject + '_camera.json'
    dp = og_p + 'Human36M_subject' + subject + '_data.json'
    jp = og_p + 'Human36M_subject' + subject + '_joint_3d.json'

    with open(cp, 'r') as CAM:
        cam = json.load(CAM)

    with open(dp, 'r') as DATA:
        data = json.load(DATA)
    
    with open(jp, 'r') as JOINT:
        joint = json.load(JOINT)

    print(' ')
    print('Loading camera parameters...')
    f = cam['2']['f']
    c = cam['2']['c']
    K = np.matrix([[f[0],   0,      c[0]],
                   [0,      f[1],   c[1]],
                   [0,      0,      1]])
    R = np.matrix(cam['2']['R'])
    t = np.array(cam['2']['t']).reshape((3,1))
    # From get_depth.py also from camera-parameters.json
    distcoeffs = np.array([-0.194213629607385, 0.240408539138292, -0.0027408943961907, -0.001619026613787, 0.00681997559022603], dtype=np.float32)
    print('Camera parameters loaded!')

    for i in tqdm(range(len(data['images']))):
        action = str(data['images'][i]['action_idx'])
        subaction = str(data['images'][i]['subaction_idx'])
        frame = str(data['images'][i]['frame_idx'])
        frame_number_png_name = str(data['images'][i]['file_name'].split('/')[-1].split('.')[0].split('_')[-1])
        
        ID = int('1' + subject.zfill(2) + action.zfill(2) + subaction.zfill(2) + frame.zfill(6) + frame_number_png_name)
        
        name = data['images'][i]['file_name']
        height = data['images'][i]['height']
        width = data['images'][i]['width']

        # Transforming the coordinates into 2d
        raw_joints = np.array(joint[action][subaction][frame])          # at this point, shape = (17, 3)
        skel = raw_joints.T

        # Project skeleton into view (this is like cv2.projectPoints)
        pt = panutils.projectPoints(skel[0:3,:], K, R, t, distcoeffs)

        ############################################################################################################
        # No need to correct the order cause that was fixed from create_mhp_gt.py
        coco = []
        for number in range(17):
            coco.append(pt[0][number])
            coco.append(pt[1][number])
            coco.append(2)
        ############################################################################################################

        # bounding box stuff
        box_coords = data['annotations'][i]['bbox']
        box_area = data['annotations'][i]['bbox'][2]*data['annotations'][i]['bbox'][3]

        im = {'file_name': name,
            'height': height,
            'width': width,
            'id': ID}
                                    
        ann = {'keypoints': coco,
            'num_keypoints': 17,        # based on amount of "2" values
            #'iscrowd': 0,                       # 0 for no, 1 for yes part of a crowd
            'image_id': ID,
            'category_id': 1,                   # Corresponding to the coco person class
            'id': ID,
            'bbox': box_coords,
            'area': box_area}
                                    
        cat = {'id':1,
            'name': 'person'}
                                    
        coco_dict["images"].append(im)
        coco_dict["annotations"].append(ann)
        coco_dict["categories"].append(cat)

print(' ')
print('Dumping the json file...')

with open(new_path, 'w') as json_file:
    json.dump(coco_dict, json_file)

print('File dumped in:          ', new_path)
