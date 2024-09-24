#This code renames files and copies them over to another folder

import os
import shutil
import json
from tqdm import tqdm

skip_total = 0

data_path = '/media/veracrypt1/gloria/Dite-HRNet/data/cmu/git/OGDATA/'

new_data_dir = input('Enter path of folder where images will be renamed and copied (no / in the end):   ')
dict_path = input('Enter path of the original dictionary (ALIGN_MOD output, .json file):   ')
newf_path = input('Enter path of the new corrected dictionary:   ')

ID = 1

with open(dict_path, 'r') as DICT:
    try:
        vital = json.load(DICT)
    except json.decoder.JSONDecodeError:
        print('The file at ', dict_path, 'does not contain valid data')

# Making a list of all the directories/trials in the provided dictionary
keys = [*vital]

for k in keys:
    skip_key = 0
    print('Adding an empty dictionary for the new kinect names in', str(k))
    # Creating a new enty called 'NEW_KNAME' = list of all the new kinect frame names
    vital[k]['NEW_KNAME']= {'1': [],
                            '2': [],
                            '3': [],
                            '4': [],
                            '5': [],
                            '6': [],
                            '7': [],
                            '8': [],
                            '9': [],
                            '10': []}
    
    # This is isolating the dictionary with the 10 kinect camera files
    corr_frames = vital[k]['KFRAMES']
    # ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    cams = [*corr_frames]
    
    # Establishing how long each list will be
    dist = len(vital[k]['TIMESTAMP'])

    print('Now looping through the values of trial:', str(k))
    for cam in cams:
        skip_cam = 0
        print('Creating copies of images from camera', str(cam))
        # Creating an empty list which will be filled with indices to delete later based on if the values were saved as 'skip'
        # This list resets everytime a new directory is selected
        #delete_later = []
        
        # Looping through each file
        for i in tqdm(range(dist)):
            file_name = corr_frames[cam][i]
            
            # Changed as of Nov 23, code will do nothing if the corresponding file name is skip
            # Before used to be
            # if file_name == 'skip:
                #delete_later.append(i)
                #print('The file needs to be skipped because there is no corresponding kinect data')
            # else:
                # do rest

            if file_name != 'skip': 
                new_kname = str(ID) + '_' + k + '_' + file_name
                #print(new_kname)
                vital[k]['NEW_KNAME'][cam].append(new_kname)

                # Now changing the path and making a copy
                old_data_dir = data_path + k + '/kinectImgs/50_' +  str(cam).zfill(2)
                old_path = os.path.join(old_data_dir, file_name)
                new_path = os.path.join(new_data_dir, new_kname)

                # This part is renaming the image and creating a copy in a new folder!!
                try:
                    #shutil.copy2(old_path, new_data_dir)
                    shutil.copy2(old_path, new_path)
                    #print(f"Copied and renamed {file} to {new_name}")
                except Exception as e:
                    print(f"Failed to copy and rename {file_name}: {e}")
                
                # Increasing ID count
                ID += 1
            
            # Adding this part to deal with kinect files that were not matched. ID not increasing but still assigning a value
            else:
                vital[k]['NEW_KNAME'][cam].append('NEWskip')
                skip_cam += 1
                skip_key += 1
                skip_total += 1
        
        print('Total of skipped frames in camera', cam, ':', skip_cam)


        # THIS PART IS MISC AS OF NOV 23: sometimes the skips (those for mismatched values) don't share the same indices in between cameras, so we can't just delete it
            # Still appending a value but named NEWskip so you can easily identify it in future codes
        # Once done looping through each file, can now delete the ones that have 'skip' in it
        # In this case, we wish to delete the data from the dictionary
        #delete_later.reverse()
        #for idx in delete_later:
        #    del(vital[k]['KFRAMES'][cam][idx])
    
    # Deleting the CMU FILE and TIMESTAMP ONCE ALL OF THE CAM VALUES ARE DELETED
    #for idx in delete_later:
    #    del(vital[k]['CMU_FILE'][idx])
    #    del(vital[k]['TIMESTAMP'][idx])
    #print('Amount of skip files in', k, ': ', len(delete_later), 'per camera deleted')

    print('Total of skipped frames in scene', k, ':', skip_key)

print('AMOUNT OF TOTAL SKIPPED FRAMES:', skip_total)
# saving it as a json file
with open(newf_path, 'w') as json_file:
    json.dump(vital, json_file)
print('JSON data has been written to: ', newf_path)