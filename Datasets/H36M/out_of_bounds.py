import json
import numpy as np
from tqdm import tqdm

mhp_root = '/media/veracrypt2/gloria/H36M/annotations_mhp/'
t = str(input('Enter data type; [1] for train, [2] for val, [3] for test:         '))

if t == '1':
    subjects = ['1', '5', '6', '7', '8']
    qp = '/media/veracrypt2/gloria/H36M/quality_train.json'
elif t == '2':
    subjects = ['9']
    qp = '/media/veracrypt2/gloria/H36M/quality_val.json'
elif t == '3':
    subjects = ['11']
    qp = '/media/veracrypt2/gloria/H36M/quality_test.json'
else:
    print('ERROR: Unrecognized data type')

with open(qp, 'r') as GT:
    quality = json.load(GT)

ids = quality['ID']

# NEED TO CLEAN UP THE 3D FILE
for subject in subjects:
    sub_ids = []
    print('*' * 150)
    print('LOOKING AT SUBJECT:          ', subject)

    # Loop through ids and add into sub_ids for specifics
    for i in range(len(ids)):
        if str(ids[i])[1:3] == subject.zfill(2):
            sub_ids.append(ids[i])
    
    cp = mhp_root + 'Human36M_subject' + subject + '_camera.json'
    dp = mhp_root + 'Human36M_subject' + subject + '_data.json'
    jp = mhp_root + 'Human36M_subject' + subject + '_joint_3d.json'

    new_cp = '/media/veracrypt2/gloria/H36M/annotations_USEME/mhp/' + 'Human36M_subject' + subject + '_camera.json'
    new_dp = '/media/veracrypt2/gloria/H36M/annotations_USEME/mhp/' + 'Human36M_subject' + subject + '_data.json'
    new_jp = '/media/veracrypt2/gloria/H36M/annotations_USEME/mhp/' + 'Human36M_subject' + subject + '_joint_3d.json'

    with open(cp, 'r') as CAM:
        cam = json.load(CAM)

    with open(dp, 'r') as DATA:
        data = json.load(DATA)
    
    with open(jp, 'r') as JOINT:
        joint = json.load(JOINT)
    
    # data and joint needs to be cleaned
    print('Deleting from joint.json...')
    img_names = []
    for id in tqdm(sub_ids):
        id = str(id)

        # ID = int('1' + subject.zfill(2) + action.zfill(2) + subaction.zfill(2) + frame.zfill(6) + frame_number_png_name)                EX: 1011102001133001134
        action = str(int(id[3:5]))
        subaction = str(int(id[5:7]))
        frame = str(int(id[7:13]))
        frame_img_name = str(int(id[13:]))     # Added this cause frame_idx = actual image name (last 6 digits) - 1

        # Keeping track of the image name so we can delete it from data.json later
        img_names.append('s_' + subject.zfill(2) + '_act_' + action.zfill(2) + '_subact_' + subaction.zfill(2) + '_ca_02_' + frame_img_name.zfill(6) + '.png')

        # Deleting from joints.json
        del joint[action][subaction][frame]
    
    # Deleting from data.json
    print('Deleting from data.json...')
    
    data_ids = {'idx': [], 'actual_id': []}
    
    for j in tqdm(range(len(data['images']))):
        #print(data['images'][j]['file_name'].split('/')[-1])
        if data['images'][j]['file_name'].split('/')[-1] in img_names:
            img_names.remove(data['images'][j]['file_name'].split('/')[-1])

            data_ids['idx'].append(j)
            data_ids['actual_id'].append(data['images'][j]['id'])

            if data['images'][j]['id'] != data['annotations'][j]['id']:
                print('ERROR: images and annotations ids do not match!!')

    if len(img_names) != 0:
        print('ERROR: not everything is gone!!!')
        input('continue?')
    
    sorted(data_ids['idx'])             # Sort in numerical order
    data_ids['idx'].reverse()           # Goind backwards so it doesn't mess with the indexing
    
    for item in data_ids['idx']:
        data['images'].remove(data['images'][item])
        data['annotations'].remove(data['annotations'][item])
    
    print(' ')
    print('Dumping JSON files')
    with open(new_cp, 'w') as A:
        json.dump(cam, A)
    
    with open(new_dp, 'w') as B:
        json.dump(data, B)
    
    with open(new_jp, 'w') as C:
        json.dump(joint, C)
