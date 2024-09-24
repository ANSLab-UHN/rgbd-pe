import numpy as np
import json
import cv2
from tqdm import tqdm

# WE WANT TO CHECK THAT THE JOINTS ARE WITHIN PREDEFINED BORDERS, IN THE DEPTH-DEFINED SECTION OF THE IMAGE
p = input('Enter path of the dite gt file:          ')
n = input('Enter name of new summary dict:          ')

with open(p, 'r') as GT:
    data = json.load(GT)

p1 = [57, 171]
p2 = [812, 170]
p3 = [60, 790]
p4 = [812, 790]

x1 = 60
x2 = 812
y1 = 171
y2 = 790

anns = data['annotations']
total = len(anns)

count = 0
bad = {'ID': [],
       'index': []}

for i in tqdm(range(total)):
    joints = np.array(anns[i]['keypoints'])
    joints = joints.reshape(-1, 3)

    c = 0

    for j in range(17):
        if joints[j][0] < x1 or joints[j][0] > x2 or joints[j][1] < 171 or joints[j][1] > 790:
            c += 1
    
    if c > 0:
        count += 1
        bad['ID'].append(anns[i]['id'])
        bad['index'].append(i)

print('Total frames out of bounds:          ', count)
print('Percentage:                          ', count/total*100)

with open(n, 'w') as json_file:
    json.dump(bad, json_file)


