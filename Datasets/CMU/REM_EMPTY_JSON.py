# This code needs to loop through all scenario .json GT files
# Return values where json files are empty
# Can use this output in the future to either delete or track the GT files that have nothing in them

import json
import os

def empty_GT_files(root):
    
    empty = {}
    
    # trials = list of all directories present in the root folder
    trials = sorted(next(os.walk(root))[1])
    
    for scene in trials:
        s = str(scene)
        empty[s] = []

        new_dir = str(scene) + '/hdPose3d_stage1_coco19'
        print(new_dir)

        for _, _, files in os.walk(new_dir):
            files = sorted(files)

            for file in files:
                if file.lower().endswith('.json'):
                    file_path = new_dir + '/' + str(file)
                    #print(file_path)
                    
                    with open(file_path, 'r') as GT:
                            try:
                                data = json.load(GT)
                                
                                if isinstance(data, dict):
                                    person = data['bodies']
                                    
                                    if len(person) == 0:
                                        #print(file, 'is empty, adding to dictionary')
                                        empty[s].append(file)
                                    #else:
                                        #print('NOT EMPTY')
                                
                                else:
                                    print('Error: the .json file provided is not a dictionary')
                            
                            except json.decoder.JSONDecodeError:
                                print('The file ', file, 'does not contain valid data')
                                empty[s].append(file)

                else:
                    print('Error: The selected file is not of JSON type')


    # by now, list of names of empty .json files has been created
    # now I want to loop through this list and delete the files in it from the main folder
    k = [*empty]
        
    for key in k:
        for i in range(len(empty[key])):
            delete_path = root + '/' + key + '/hdPose3d_stage1_coco19/' + str(empty[key][i])

            try:
                if os.path.exists(delete_path):
                    os.remove(delete_path)
                    print('Deleted')
                else:
                    print('Error: The file ', delete_path, 'does not exist')
            except OSError as e:
                print('Error: ', e, '.Failed to delete the file' )



path = '/media/veracrypt1/gloria/Dite-HRNet/data/cmu/git/SKELETON' 
empty_GT_files(path)