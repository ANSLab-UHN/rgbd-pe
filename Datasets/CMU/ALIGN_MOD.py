# What this code does:
# MAKE SURE TO CHANGE THE ROOT FOLDER ACCORDINGLY
# This code creates a dictionary with the following information:
    # 'CMU_FILE': list of corresponding original hd3dPose annotation or GT file
    # 'TIMESTAMP': list of univTime value from each hd GT .json file
    # 'KFRAMES': corresponding number of the kinect frame (from ksynctables) based on univTime
        # {1: [] until 10: []}, one for each kinect camera
# In the end, the code:
    # writes this dictionary to a JSON file (change the name accordingly below)
    # prints how many frames had to be skipped in total
    # prints a summary of how many frames were skipped per scenario or directory

from FUNCTIONS import take_closest
import json
import os
from tqdm import tqdm

root =      input('Enter root path:           ')
gt_root =   input('Enter gt files path:       ')
newf_path = input('Enter new files path:      ')

vital = {}
trials = sorted(next(os.walk(gt_root))[1])
skip_total = 0
mismatch_total = 0
skip_data = {}
mismatch_data = {}

for scene in trials:
    s = str(scene)
    skip_num = 0
    
    # Creating an empty dictionary for each scene, will have summary information about how many kinect frames were mismatched to GT and marked as 'skip'
    mismatch_data[s] = {'1': [], 
                        '2': [], 
                        '3': [], 
                        '4': [], 
                        '5': [], 
                        '6': [], 
                        '7': [], 
                        '8': [], 
                        '9': [], 
                        '10': []}
    
    gt_path = gt_root + '/' + s + '/hdPose3d_stage1_coco19/'
    print('Looking at ground truth path', gt_path)
    vital[s] = {'CMU_FILE': [],
                'TIMESTAMP': [], 
                'KFRAMES': {'1': [], 
                            '2': [], 
                            '3': [], 
                            '4': [], 
                            '5': [], 
                            '6': [], 
                            '7': [], 
                            '8': [], 
                            '9': [], 
                            '10': []}}
    print('Dictionary for ', s, 'created')

    files = sorted(next(os.walk(gt_path))[2])

    # Looping through all the GT annotation files, one by one
    for file in tqdm(files):

            # Checking that the path is correct
            if file.lower().endswith('.json'):
                gtfile_path = gt_path + '/' + str(file)
                #print(file)
                vital[s]['CMU_FILE'].append(file)

                # Opening the file and saving it as data_gt
                with open(gtfile_path, 'r') as GT:
                    try:
                        data_gt = json.load(GT)
                    except json.decoder.JSONDecodeError:
                        print('The file at ', gtfile_path, 'does not contain valid data')
                    
                # Saving the timestamp into the big dictionary with all of the information
                timestamp = data_gt['univTime']
                vital[s]['TIMESTAMP'].append(timestamp)
                # At this point the dictionary timestamp values are all filled up

            else:
                print('ERROR: path is not poiting to a .JSON file')
    

# Now we want to loop through the timestamp values in the dictionary while looking at kframes
# Need to loop through all 10 kinect cameras for each scene   
    print('SUCCESS: All timestamp values added into the dictionary')
    ksync_path = root + '/OGDATA/' + s + '/ksynctables_' + s + '.json'

    # Looking at each key (which is a directory = scenario) and looping through all of the timestamp values
    print('Using synctables to find the corresponding kinect images...')
    for j in range(10):
        cam_num = str(j + 1)
        node = 'KINECTNODE' + str(cam_num)
        
        print('Looking at camera', node, '    SCENE:', s)
        mismatch_num = 0
        # Range 10 for each kinect camera
        for i in tqdm(range(len(vital[s]['TIMESTAMP']))):
###########################################################################################################################################
# WATCH OGDATA FILE PATH HERE
            ksync_path = root + '/OGDATA/'+ s + '/ksynctables_' + s + '.json'
            #print("LOOKING AT ksync table path: ", ksync_path)

            with open(ksync_path, 'r') as KSYNC:
                try:
                    data_ksync = json.load(KSYNC)
                except json.decoder.JSONDecodeError:
                    print('The file at ', ksync_path, 'does not contain valid data')
            
            k_sync_list = data_ksync['kinect']['color'][node]['univ_time']

###########################################################################################################################################
            while k_sync_list[-1] == -1.0:
                del(k_sync_list[-1])

            # This part is added to compensate for the fact that the starting ksync value may be higher than the one we want to match
            idx  =  0
            while k_sync_list[idx] == -1.0:
                idx += 1
            starting_val = k_sync_list[idx]
            if starting_val <= vital[s]['TIMESTAMP'][i]:
###########################################################################################################################################
                matched_k = take_closest(k_sync_list, vital[s]['TIMESTAMP'][i])
                diff = int(vital[s]['TIMESTAMP'][i] - matched_k)
                if diff >= 36:
                    # If the difference is too high, no new kinect frame will be matched
                    vital[s]['KFRAMES'][cam_num].append('skip')
                    mismatch_num += 1
                    mismatch_total += 1
                else:
                    corr_kframe = k_sync_list.index(matched_k) + 1
                    cam = '50_' + cam_num.zfill(2)
                    frame_num = str(corr_kframe).zfill(8)
                    corr_kimg = cam + '_' + frame_num + '.jpg'
                    vital[s]['KFRAMES'][cam_num].append(corr_kimg)
            
            # In the case that the univ time is smaller than the first value in the ksync table, corresponding kinect img will be marked as 'skip'
            else:
                vital[s]['KFRAMES'][cam_num].append('skip')
                #print('GT FRAME SKIPPED')
                skip_num += 1
                skip_total += 1
    
        mismatch_data[s][cam_num] = mismatch_num
    skip_data[s] = skip_num

# saving it as a json file
with open(newf_path, 'w') as json_file:
    json.dump(vital, json_file)
print('JSON data has been written to: ', newf_path)

print(skip_total, 'frames skipped in total')
print('Summary of skipped data')
print(skip_data)

print(mismatch_total, 'frames mismatched in total')
print('Summary of mismatched data')
print(mismatch_data)