# Modified version of convert_anns.py
# Code personalized, using input from the dictionary rather than looping through folder like before

import os
import numpy as np
import json
import panutils
import matplotlib.pyplot as plt
plt.rcParams['image.interpolation'] = 'nearest'
from tqdm import tqdm

#######################################################################################################################
kcalib_path = '/media/veracrypt1/gloria/Dite-HRNet/data/OFFICIAL_CMU/OGDATA'
imgs_path = '/media/veracrypt1/gloria/Dite-HRNet/data/OFFICIAL_CMU/imgs_test'
gt_path = '/media/veracrypt1/gloria/Dite-HRNet/data/OFFICIAL_CMU/SKELETON_TEST'

# Path where the correct values are located (output of ID_AND_RENAME) dictionary
correct_path = '/media/veracrypt1/gloria/Dite-HRNet/data/cmu/git/SKELETON/CLUSTER_TEST_corrected.json'
yolo_path = '/media/veracrypt1/gloria/Dite-HRNet/YOLO_OFFICIAL_fullTEST.json'

# Path where the resulting big coco dict is written into
file_path = 'CLUSTER_COCOanns_test.json'

# Adding for where the corrected yolo dictionary is going (only one bounding box per image id, the one with the highest score is kept)
box_path = 'CLUSTER_corrected_BBOX_test.json'
#######################################################################################################################

# First let's go through the yolo information and create a variable with the yolo list
with open(yolo_path, 'r') as YOLO:
    try:
        yolo_data = json.load(YOLO)
    except json.decoder.JSONDecodeError:
        print('The file at ', yolo_path, 'does not contain valid data')

# This is a list of all of the image_ids present in the yolo json file
yolo_id_list = []

short_yolo = []

# Creating the variable that has all of the yolo stuff
# yolo_data has the whole list of dictionaries of complete annotations
# yolo_id_list is just a list of the image_ids that have a yolo bbox label
for i in range(len(yolo_data)):
    yolo_id_list.append(yolo_data[i]['image_id'])

# This code needs to loop through the images depending on the GT label
# Creating an empty dictionary that respects the COCO format
cmu_to_coco_dict = {"images": [], "annotations": [], "categories": []}

with open(correct_path, 'r') as DICT:
    try:
        info = json.load(DICT)
    except json.decoder.JSONDecodeError:
        print('The file at ', correct_path, 'does not contain valid data')

# This is getting the list of scenarios directly from what is in the dictionary
seq_names = [*info] 
for seq_name in seq_names:
    print('Creating labels for trial:', seq_name)
    
    hd_skel_json_path = gt_path + '/' + seq_name + '/hdPose3d_stage1_coco19/'

    print('Opening kcalibration file at:', kcalib_path+'/'+seq_name)
    with open(kcalib_path+'/'+seq_name+'/calibration_{0}.json'.format(seq_name)) as cfile:
        calib = json.load(cfile)
    cameras = {(cam['panel'],cam['node']):cam for cam in calib['cameras']}

    # Convert data into numpy arrays for convenience
    for k,cam in cameras.items():    
        cam['K'] = np.matrix(cam['K'])
        cam['distCoef'] = np.array(cam['distCoef'])
        cam['R'] = np.matrix(cam['R'])
        cam['t'] = np.array(cam['t']).reshape((3,1))
    
    print('Camera arrays converted from the calibration file    (Camera intrinsic and extrinsic parameters loaded)')

######################################################################################################################
# Dictionary looks like this: 
    # {pose1 = {'CMU_FILE': [],
    #           'TIMESTAMP': [], 
    #           'KFRAMES': {'1': [], 
    #                       '2': [], 
    #                       '3': [], 
    #                       '4': [], 
    #                       '5': [], 
    #                       '6': [], 
    #                       '7': [], 
    #                       '8': [], 
    #                       '9': [], 
    #                       '10': []},
    #           'NEW_KNAME': {'1': [], 
    #                         '2': [], 
    #                         '3': [], 
    #                         '4': [], 
    #                         '5': [], 
    #                         '6': [], 
    #                         '7': [], 
    #                         '8': [], 
    #                         '9': [], 
    #                         '10': []}}
    # {pose 2: .......................}

    # Looping through all 10 kinect cameras
    for cam_index in range(10):
        cam_index = cam_index + 1
        cam = cameras[(50, cam_index)]

        # Looping through all of the possible frames from the dictionray generated in previous code
        # Looping through the CMU_file length cause this is what has all of the frames
        for i in tqdm(range(len(info[seq_name]['CMU_FILE']))):
            # We only want this to run if the file isn't skip, or else it's not worth it
            NAME = info[seq_name]['NEW_KNAME'][str(cam_index)][i]
            
            if NAME != 'NEWskip':
                # This is pulling the id value directly from what is right in front of the image name
                ID = int(NAME.split('_', 1)[0].replace('.', '').upper())
                
                # Adding this part here to check if yolo exists
                if yolo_id_list.count(ID) >= 1:
                
                    skel_frame_name = info[seq_name]['CMU_FILE'][i]
                    
                    try:
                        # Load the json file with this frame's skeletons
                        skel_json_fname = hd_skel_json_path + skel_frame_name

                        with open(skel_json_fname) as dfile:
                            bframe = json.load(dfile)

                        # Cycle through all detected bodies
                            # AS OF NOV 25, ONLY DOING SINGLE BODY STUFF
                        for body in bframe['bodies']:
                            
                            # There are 19 3D joints, stored as an array [x1,y1,z1,c1,x2,y2,z2,c2,...]
                            # where c1 ... c19 are per-joint detection confidences
                            skel = np.array(body['joints19']).reshape((-1,4)).transpose()

                            # Project skeleton into view (this is like cv2.projectPoints)
                            pt = panutils.projectPoints(skel[0:3,:],
                                        cam['K'], cam['R'], cam['t'], 
                                        cam['distCoef'])
                            
            #######################################################################################################################
                            coco = []
                            # Creating this correct list because the cmu keypoint list doesn't follow the same order as the coco one
                            # Following ording from:
                                # CMU: https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox/tree/master
                                # COCO: https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#coco-dataset-format
                            correct = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]

                            # This is looping through the pts values (in pixels) and appending them IN THE CORRECT ORDER to the coco list
                                # coco = corrected annotations, pixel values in the correct order. THIS IS WHAT I NEED TO USE TO MAKE THE JSON FILE.
                            
                            num_keypts = 0

                            for i, c in enumerate(correct):
                                coco.append(pt[0][c])
                                coco.append(pt[1][c])
                                if pt[0][c] and pt[1][c] == 0:
                                    coco.append(0)
                                else:
                                    coco.append(2)          # AS OF NOW ONLY LABELLING THE THIRD VALUES
                                    num_keypts += 1

                            height = 1080
                            width = 1920

                            #camera_name = '00_' + str(cam_index).zfill(2)
                            #frame_num = str(hd_idx).zfill(8)
                            #NAME = camera_name + '_' + frame_num + '.jpg'

                            if yolo_id_list.count(ID) == 1:
                                idx = yolo_id_list.index(ID)
                                box_coords = yolo_data[idx]['bbox']
                                box_area = yolo_data[idx]['bbox'][2]*yolo_data[idx]['bbox'][3]
                                
                                # Adding this part of the code to append and create the shortened yolo file that doesn't have bounding box duplicates
                                y_dict = {}
                                y_dict['bbox'] = box_coords
                                y_dict['category_id'] = 1
                                y_dict['image_id'] = ID
                                y_dict['score'] = yolo_data[idx]['score']
                                short_yolo.append(y_dict)

                            if yolo_id_list.count(ID) > 1:
                                #print('More than 1 yolo bbox detected in image id:', ID)
                                # This makes a list of the indices at which the ID is present
                                appear = [index for (index, item) in enumerate(yolo_id_list) if item == ID]
                                conf_list = []
                                # now looping through all the times more than one person is detected and selecting the time with the highest confidence score
                                for id_index in appear:
                                    #refer back to big yolo dict
                                    conf_list.append(yolo_data[id_index]['score'])
                                max_conf = max(conf_list)
                                max_id = conf_list.index(max_conf)
                                box_coords = yolo_data[max_id]['bbox']
                                box_area = (yolo_data[max_id]['bbox'][2])*(yolo_data[max_id]['bbox'][3])

                                # Adding this part of the code to append and create the shortened yolo file that doesn't have bounding box duplicates
                                y_dict = {}
                                y_dict['bbox'] = box_coords
                                y_dict['category_id'] = 1
                                y_dict['image_id'] = ID
                                y_dict['score'] = yolo_data[max_id]['score']
                                short_yolo.append(y_dict)

                            # Adding to the dictionary:
                            im = {
                                'file_name': NAME,
                                'height': height,
                                'width': width,
                                'id': ID
                                }
                            
                            ann = {
                                'keypoints': coco,
                                'num_keypoints': num_keypts,        # based on amount of "2" values
                                #'iscrowd': 0,                       # 0 for no, 1 for yes part of a crowd
                                'image_id': ID,
                                'category_id': 1,                   # Corresponding to the coco person class
                                'id': ID,
    ###############################################################################################################################
    # Last part left is finding a way to implement the bbox findings                         
                                'bbox': box_coords,
                                'area': box_area
    ###############################################################################################################################
                                }
                            
                            cat = {
                                'id':1,
                                'name': 'person'
                                }
                            
                            cmu_to_coco_dict["images"].append(im)
                            cmu_to_coco_dict["annotations"].append(ann)
                            cmu_to_coco_dict["categories"].append(cat)
    #######################################################################################################################

                    except IOError as e:
                        print('Error reading {0}\n'.format(skel_json_fname)+e.strerror)

    # END OF CAMERA CALIB CODE TO GET PIXEL VALUES

######################################################################################################################
# IN THE END OF EVERYTHING, WE WANNA WRITE THIS BIG DICTIONARY TO A .JSON FILE
with open(file_path, 'w') as json_file:
    json.dump(cmu_to_coco_dict, json_file)

with open(box_path, 'w') as yolo_file:
    json.dump(short_yolo, yolo_file)

print('Coco-equivalent JSON data has been written to: ', file_path)
print('Cropped yolo bbox data has been written to: ', box_path)