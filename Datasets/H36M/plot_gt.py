import json
import os
import random
import cv2
import numpy as np
from tqdm import tqdm

#imgs_path = str(input('Enter images folder path:            '))
imgs_path = '/media/veracrypt2/gloria/H36M/h36m_rgbd'
gt_path = str(input('Enter GT file path:                  '))
#out_folder = str(input('Enter output path:                   '))
out_folder = '/media/veracrypt2/gloria/H36M/CHECKME'
amt = int(input('Enter amount of files to check:      '))

with open(gt_path, 'r') as GT:
    try:
        data = json.load(GT)
    except json.decoder.JSONDecodeError:
        print('The file at ', gt_path, 'does not contain valid data')

imgs= data['images']
anns = data['annotations']
cats = data['categories']

# FIRST CHECK THAT ALL KEYS HAVE THE SAME LENGTHS
if len(imgs) == len(anns) == len(cats):
    print('SUCCESS! All three keys have lists of equal lengths')
    total = len(imgs)
    print('Lengths:        ', total)
else:
    print('ERROR: NOT ALL KEYS HAVE THE SAME LENGTH')
    print('Images:         ', len(imgs))
    print('Annotations:    ', len(anns))
    print('Categories:     ', len(cats))

numbers = random.sample(range(total), amt)
#print('Indices of files being checked:  ', numbers)

for _, i in tqdm(enumerate(numbers)):
    n = imgs[i]['file_name']
    k = anns[i]['keypoints']
    b = anns[i]['bbox']

    #print('Looking at the image:    ', n)

    pimage = os.path.join(imgs_path, n)
    newname = 'GTandBBOX_'+ n.split('/')[-1]
    pout = os.path.join(out_folder, newname)
    #print('New images being saved as:   ', pout)

    # Reading the image
    image = cv2.imread(pimage, cv2.IMREAD_UNCHANGED)
    
    # Reshape keypoints into an array of shape (num_keypoints, 3)
    keypoints = np.array(k).reshape(-1, 3)

    lines = [(5,6), (6,8), (8,10), (5,7), (7,9), (5,11), (6,12), (11,13), (13,15), (12,14), (14,16), (11,12)]
    i = 0

    # Plotting keypoints and lines
    for x, y, score in keypoints:
        cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255, 255), -1)
        cv2.putText(image, str(i+1), (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0, 255), 2)
        i += 1

    for line in lines:
            start_keypoint = keypoints[line[0]]
            end_keypoint = keypoints[line[1]]
            cv2.line(image, (int(start_keypoint[0]), int(start_keypoint[1])),
                    (int(end_keypoint[0]), int(end_keypoint[1])), (0, 0, 255, 255), 2)
    

    # Plotting the bounding box (0 = top left coords, 1 = bottom right coords)
    x0 = int(b[0])
    y0 = int(b[1])
    x1 = x0 + int(b[2])
    y1 = y0 + int(b[3])

    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0, 255), 2)

    # Write the file
    cv2.imwrite(pout, image)
