# Code takes untouched h36m joint json dict (raw gt with wrong spots) and fixes it so everything is nice and in coco format

import os
import json
import numpy as np
from tqdm import tqdm

subjects = ['1','5','6','7','8','9','11']

for subject in subjects:
    print('*' * 150)
    print('LOOKING AT SUBJECT:          ', subject)

    p = '/media/veracrypt2/gloria/H36M/annotations/Human36M_subject' + subject + '_joint_3d.json'
    new_path = '/media/veracrypt2/gloria/H36M/annotations_mhp/Human36M_subject' + subject + '_joint_3d.json'

    with open(p, 'r') as GT:
        data = json.load(GT)

    # Loop through the whole thing...
    actions = data.keys()

    for action in tqdm(actions):
        subactions = data[action].keys()

        for subaction in subactions:
            frames = data[action][subaction].keys()

            for frame in frames:
                joints = []
                raw = data[action][subaction][frame]
                neck = np.array(raw[8])
                nose = np.array(raw[9])
                top_of_head = np.array(raw[10])
                ls = np.array(raw[11])
                rs = np.array(raw[14])

                # INTERPOLATING STUFF
                shoulders = np.linalg.norm(ls - rs)
                head_height = shoulders             
                head_width = (2*head_height)/3
                eye_width = head_width/5

                V = top_of_head - neck              # Vertical vector (V)
                eye_height = neck + 0.8 * V         # Eye height (>halfway between neck and top of the head)
                nose_vector = nose - neck           # Horizontal vector (H) perpendicular to V and passing through the nose
                H = np.cross(V, nose_vector)        # One way to find H is to use cross product with another vector, such as nose-neck vector
                H = H / np.linalg.norm(H)           # Normalize H
                eye_distance = eye_width            # Define the eye distance from the nose - from anatomny links

                # Compute the positions of the eyes
                left_eye = eye_height + eye_distance * H
                right_eye = eye_height - eye_distance * H

                # Ears are at the same height as eyes but further laterally
                ear_distance = 2.5*(eye_width)      # Define ear distance from nose
                left_ear = eye_height + ear_distance * H
                right_ear = eye_height - ear_distance * H


                # NOW WE CAN PUT IT ALL TOGETHER :)
                joints.append(nose.tolist())
                joints.append(left_eye.tolist())
                joints.append(right_eye.tolist())
                joints.append(left_ear.tolist())
                joints.append(right_ear.tolist())
                joints.append(ls.tolist())     # left shoulder
                joints.append(rs.tolist())     # right shoulder
                joints.append(raw[12])         # left elbow
                joints.append(raw[15])         # right elbow
                joints.append(raw[13])         # left wrist
                joints.append(raw[16])         # right wrist
                joints.append(raw[4])          # left hip
                joints.append(raw[1])          # right hip
                joints.append(raw[5])          # left knee
                joints.append(raw[2])          # right knee
                joints.append(raw[6])          # left ankle
                joints.append(raw[3])          # right ankle

                data[action][subaction][frame] = joints

    print(' ')
    print('Dumping the json file...')
    with open(new_path, 'w') as json_file:
        json.dump(data, json_file)

