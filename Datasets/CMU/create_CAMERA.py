import os
import json
from tqdm import tqdm

#p = '/media/veracrypt1/gloria/Dite-HRNet/data/OFFICIAL_CMU/OGDATA'
p = '/media/veracrypt1/gloria/Dite-HRNet/data/DEEP_RED_1K_CMU/OGDATA'
trials= sorted(os.listdir(p))

c = 1
corr = {'trial': [],
        'number': []}

for t in trials:
    #n = '/media/veracrypt1/gloria/MobileHumanPose/data/CMU/FULL_anns/CMU_subject' + str(c) + '_camera.json'
    n = '/media/veracrypt1/gloria/MobileHumanPose/data/CMU/1k_anns/CMU_subject' + str(c) + '_camera.json'
    output = {'1': {'R': [],
                    't': [],
                    'f': [],
                    'c': []},
              '2': {'R': [],
                    't': [],
                    'f': [],
                    'c': []},
              '3': {'R': [],
                    't': [],
                    'f': [],
                    'c': []},
              '4': {'R': [],
                    't': [],
                    'f': [],
                    'c': []},
              '5': {'R': [],
                    't': [],
                    'f': [],
                    'c': []},
              '6': {'R': [],
                    't': [],
                    'f': [],
                    'c': []},
              '7': {'R': [],
                    't': [],
                    'f': [],
                    'c': []},
              '8': {'R': [],
                    't': [],
                    'f': [],
                    'c': []},
              '9': {'R': [],
                    't': [],
                    'f': [],
                    'c': []},
              '10': {'R': [],
                    't': [],
                    'f': [],
                    'c': []}}
    
    corr['trial'].append(t)
    corr['number'].append(c)

    source = p + '/' + str(t) +'/calibration_' + str(t) + '.json'
    print('LOOKING AT THE FILE:     ', source)

    with open(source, 'r') as GT:
        data = json.load(GT)

    d = data['cameras']

    for i in tqdm(range(len(d))):
        if d[i]['name'].split('_')[0] == '50':
            cam_id = d[i]['name'].split('_')[1]
            if cam_id != '10':
                cam_id = cam_id[-1]

            output[cam_id]['R'] = d[i]['R']

            output[cam_id]['t'].append(d[i]['t'][0][0])
            output[cam_id]['t'].append(d[i]['t'][1][0])
            output[cam_id]['t'].append(d[i]['t'][2][0])

            output[cam_id]['f'].append(d[i]['K'][0][0])
            output[cam_id]['f'].append(d[i]['K'][1][1])

            output[cam_id]['c'].append(d[i]['K'][0][2])
            output[cam_id]['c'].append(d[i]['K'][1][2])

    c += 1
    
    with open(n, 'w') as json_file:
        json.dump(output, json_file)
    print('DUMPED DICTIONARY INTO:      ', n)

print(corr)