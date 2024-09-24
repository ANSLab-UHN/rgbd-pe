# Code takes original annotations file (as downloaded) in the annotations_ORIGINAL folder and removes the GT for the files that don't exist with depth

import os
import json
from tqdm import tqdm

subjects = ['1', '5', '6', '7', '8', '9', '11']
new_root = '/media/veracrypt2/gloria/H36M/annotations/'

# This is a list of the folders of the data created = should correspond to the order in which they appear in the data dictionray GT
trials = sorted(os.listdir('/media/veracrypt2/gloria/H36M/h36m_rgbd'))

for subject in subjects:
    print('*' * 150)
    print('LOOKING AT SUBJECT:          ', subject)
    s = subject.zfill(2)
    
    # Create a cropped trial list based on the subject
    sub_trials = []
    for t in trials:
        if t.split('_')[1] == s:
            sub_trials.append(t)
    # At this point, we have sub_trials = list of the folders specific to the subject only
    
    dp = '/media/veracrypt2/gloria/H36M/annotations_ORIGINAL/Human36M_subject' + subject + '_data.json'
    jp = '/media/veracrypt2/gloria/H36M/annotations_ORIGINAL/Human36M_subject' + subject + '_joint_3d.json'

    new_dp = new_root + 'Human36M_subject' + subject + '_data.json'
    new_jp = new_root + 'Human36M_subject' + subject + '_joint_3d.json'

    new_d = {'images': [],
             'annotations': []}
    new_j = {}


    with open(dp, 'r') as DATA:
        data = json.load(DATA)
    
    with open(jp, 'r') as JOINT:
        joint = json.load(JOINT)
    
    print('Making list of folders from GT...')
    specific = []
    for i in range(len(data['images'])):
        want = data['images'][i]['file_name']
        if want.split('_')[-2] == '02' and want.split('/')[0] not in specific:
            specific.append(want.split('/')[0])
    
    print(' ')
    print('Checking that the GT folders correspond to the image folders...')
    # Or that at least the elements are present in the list
    small_set = set(sub_trials)
    large_set = set(specific)
    is_subset = small_set.issubset(large_set)
  
    if is_subset == True:
        print('         GOOD!')
    else:
        print('         ERROR: folders do not correspond')
        input('THIS DID NOT WORK!!!!!!!!!!!!!!!!!!!!!!!!')
    
    print(' ')
    print('Looping through each folder...')
    imgs = []
    for img_folder in sub_trials:
        print('LOOKING AT:          ', img_folder)
        # This is the list of images that are in the folder
        imgs = sorted(os.listdir('/media/veracrypt2/gloria/H36M/h36m_rgbd/' + img_folder))
        for k in range(len(imgs)):
            imgs[k] = imgs[k].split('.')[0]
    
        for j in tqdm(range(len(data['images']))):
            # First let's make sure that this is for camera 2 only...
            if data['images'][j]['file_name'].split('_')[-2] == '02':
                img_name = data['images'][j]['file_name'].split('/')[1].split('.')[0]       # split by period so we get rid of the file extension since they're different
                
                if img_name in imgs:
                    imgs.remove(img_name)       # make it computationally easier over time

                    new_d['images'].append(data['images'][j])
                    new_d['annotations'].append(data['annotations'][j])

                    action = str(data['images'][j]['action_idx'])
                    subaction = str(data['images'][j]['subaction_idx'])
                    frame = str(data['images'][j]['frame_idx'])

                    if action not in new_j:
                        new_j[action] = {}
                    if subaction not in new_j[action]:
                        new_j[action][subaction] = {}

                    new_j[action][subaction][frame] = joint[action][subaction][frame]

    print(' ')
    print('Replacing the file names with the png extension...')
    for g in range(len(new_d['images'])):
        new_d['images'][g]['file_name'] = new_d['images'][g]['file_name'].split('.')[0] + '.png'

    print(' ')
    print('Dumping json files...')

    with open(new_dp, 'w') as NEW_DATA:
        json.dump(new_d, NEW_DATA)
    
    with open(new_jp, 'w') as NEW_JOINT:
        json.dump(new_j, NEW_JOINT)

    print('New data file:            ', new_dp)
    print('New joint file:           ', new_jp)
    print(' ')
    print(' ')
    print(' ')
