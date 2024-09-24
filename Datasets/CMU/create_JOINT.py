import os
import json
import numpy as np
from tqdm import tqdm

a = input('Enter aligned depth dictionary path:             ')
t = input('Enter dataset type:  a for 1k data, b for full:  ')

if t == 'a':
    corr = {'trial': ['161029_piano2', '161029_piano4', '171026_pose3'], 'number': [1, 2, 3]}
    ids = {'161029_piano4': [1, 530], '171026_pose3': [531, 1250]}
    n_root = '/media/veracrypt1/gloria/MobileHumanPose/data/CMU/1k_anns/CMU_subject'
elif t == 'b':
    corr = {'trial': ['161029_flute1', '161029_piano2', '161029_piano3', '161029_piano4', '170307_dance5', '170407_office2', '170915_office1', '171026_cello3', '171026_pose1', '171026_pose2', '171026_pose3', '171204_pose1', '171204_pose2', '171204_pose3', '171204_pose4', '171204_pose5', '171204_pose6'], 'number': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]}
    #ids = {'161029_flute1': [1, 115264], '161029_piano2': [115265, 132520], '161029_piano4': [132521, 185749], '171026_cello3': [349924, 365963], '171026_pose3': [504874, 576630], '171204_pose1': [576631, 842090], '171204_pose2': [842091, 948020], '171204_pose4': [1033320, 1303635], '171204_pose5': [1303636, 1555263], '161029_piano3': [2000000, 2044225], '171026_pose1': [2044226, 2259345], '170407_office2': [3000000, 3046978], '171026_pose2': [365982, 504873], '170915_office1': [297299, 349904], '171204_pose3': [948022, 1033319]}
    #ids = {'171026_pose2': [365982, 504873], '170915_office1': [297299, 349904], '171204_pose3': [948022, 1033319]}
    ids = {'170307_dance5': [185750, 297286], '171204_pose6': [3046979, 3261734]}
    n_root = '/media/veracrypt1/gloria/MobileHumanPose/data/CMU/FULL_anns/CMU_subject'
else:
    print('ERROR: Unrecognized data type value')


print('Opening the aligned depth file...')
with open(a, 'r') as GT:
    data = json.load(GT)

# Aligned depth dictionary is organized like this:
    # data.keys() = trial names
        # Each trial {} with keys:
            # CMU_FILE      - original GT file with world coords
            # TIMESTAMP     - corresponding univTime
            # KFRAMES       - old name of the color frame
                # dict with 10 keys
            # NEW_KNAME     - new name of the color frame (with id at beginning)
                # dict with 10 keys
            # KDEPTH        - index of corresponding depth data
                # dict with 10 keys

for subject in data.keys():
    print('LOOKING AT TRIAL:                ', subject)
    ind = corr['trial'].index(subject)
    c = corr['number'][ind]
    print('CORRESPONDING SUBJECT NUMBER:    ', c)
    
    n = n_root + str(c) + '_joint_3d.json'

    info = {"2": {"1": {}}}

    # Need this to loop through all of the cameras (sub dictionaries)
    count = 0

    for j in range(10):
        cam_num = str(j+1)
        print('     LOOKING AT CAMERA:      ', cam_num)

        if len(data[subject]['CMU_FILE']) != len(data[subject]['NEW_KNAME'][cam_num]) != len(data[subject]['KDEPTH'][cam_num]):
            print('ERROR: LENGHTS NOT EQUAL')

        for i in tqdm(range(len(data[subject]['NEW_KNAME'][cam_num]))):
            if data[subject]['NEW_KNAME'][cam_num][i] != 'NEWskip' and data[subject]['KDEPTH'][cam_num][i] not in ['no', 'dskip']:
                # Getting the corresponding ID
                idx = data[subject]['NEW_KNAME'][cam_num][i].split('_')[0]
                # Getting the corresponding 3D data
                filename = '/media/veracrypt1/gloria/Dite-HRNet/data/cmu/git/OGSKELETON/' + subject + '/hdPose3d_stage1_coco19/' + data[subject]['CMU_FILE'][i]

                with open(filename, 'r') as OK:
                    gt = json.load(OK)
                
                if len(gt['bodies']) == 1:
                    raw_skel = gt['bodies'][0]['joints19']

                    result = [raw_skel[i:i+3] for i in range(0, len(raw_skel), 4) if i % 4 != 3]

                    coco = []
                    # Creating this correct list because the cmu keypoint list doesn't follow the same order as the coco one
                    # Following order from:
                        # CMU: https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox/tree/master
                        # COCO: https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#coco-dataset-format
                    correct = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]

                    for _, c in enumerate(correct):
                        coco.append(result[c])
                    
                    info['2']['1'][str(idx)] = coco

                else:
                    print('ERROR: more than 1 body in the ground truth')
            
            else:
                count += 1
    
    print('TOTAL FRAMES SKIPPED:     ', count)


    with open(n, 'w') as json_file:
        json.dump(info, json_file)
    print('FILE DUMPED AS:      ', n)