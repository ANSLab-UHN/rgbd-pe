import os
import json
from tqdm import tqdm

corr = {'trial': ['161029_flute1', '161029_piano2', '161029_piano3', '161029_piano4', '170307_dance5', '170407_office2', '170915_office1', '171026_cello3', '171026_pose1', '171026_pose2', '171026_pose3', '171204_pose1', '171204_pose2', '171204_pose3', '171204_pose4', '171204_pose5', '171204_pose6'], 'number': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]}
#ids = {'161029_flute1': [1, 115264], '161029_piano2': [115265, 132520], '161029_piano4': [132521, 185749], '171026_cello3': [349924, 365963], '171026_pose3': [504874, 576630], '171204_pose1': [576631, 842090], '171204_pose2': [842091, 948020], '171204_pose4': [1033320, 1303635], '171204_pose5': [1303636, 1555263], '161029_piano3': [2000000, 2044225], '171026_pose1': [2044226, 2259345], '170407_office2': [3000000, 3046978]}
#ids = {'171026_pose2': [365982, 504873], '170915_office1': [297299, 349904], '171204_pose3': [948022, 1033319]}
ids = {'170307_dance5': [185750, 297286], '171204_pose6': [3046979, 3261734]}


#corr = {'trial': ['161029_piano2', '161029_piano4', '171026_pose3'], 'number': [1, 2, 3]}
#ids = {'161029_piano4': [1, 530], '171026_pose3': [531, 1250]}
#ids = {'161029_piano2': [1, 170]}

print('REMINDER: ARE THE FILE PATHS CORRECT?????')
print('         CHECK LINES 5 - 14 and 35 - 37')

g = input('Enter path of the GT file:       ')
# /media/veracrypt1/gloria/Dite-HRNet/data/DEEP_RED_1K_CMU/annotations/DR_1k_COCOanns_train.json

print('Opening the GT annotations file...')
with open(g, 'r') as GT:
    data = json.load(GT)

vis = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]

imgs = data['images']
anns = data['annotations']

for subject in ids.keys():
    print('LOOKING AT TRIAL:                ', subject)
    ind = corr['trial'].index(subject)
    c = corr['number'][ind]
    print('CORRESPONDING SUBJECT NUMBER:    ', c)
    
    n = '/media/veracrypt1/gloria/MobileHumanPose/data/CMU/FULL_anns/CMU_subject' + str(c) + '_data.json'
    #n = '/media/veracrypt1/gloria/MobileHumanPose/data/CMU/1k_anns/CMU_subject' + str(c) + '_data.json'

    info = {'images': [],
            'annotations': []}

    for i in tqdm(range(len(imgs))):
        in_im = {}
        in_ann = {}

        mini = ids[subject][0]
        maxi = ids[subject][1]

        if mini <= imgs[i]['id'] <= maxi:
            in_im['id'] = imgs[i]['id']
            in_im['file_name'] = imgs[i]['file_name']
            in_im['width'] = 1920
            in_im['height'] = 1080
            in_im['subject'] = c
            in_im['action_name'] = "Directions"
            in_im['action_idx'] = 2
            in_im['subaction_idx'] = 1
            in_im['cam_idx'] = int(imgs[i]['file_name'].split('_')[4])
            in_im['frame_idx'] = imgs[i]['id']

            in_ann['id'] = imgs[i]['id']
            in_ann['image_id'] = imgs[i]['id']
            in_ann['keypoints_vis'] = vis
            in_ann['bbox'] = anns[i]['bbox']
        
            info['images'].append(in_im)
            info['annotations'].append(in_ann)
    
    
    with open(n, 'w') as json_file:
        json.dump(info, json_file)
    print('FILE DUMPED AS:      ', n)