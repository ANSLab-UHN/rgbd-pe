import os
from tqdm import tqdm
import numpy as np
import json
import cdflib
import cv2
from PIL import Image

#new_p = input('Enter path where new RGBD images are created:        ')
#new_16 = input('Enter path where uint16 depth images will go:        ')
new_p = '/media/veracrypt2/gloria/H36M/h36m_rgbd'
new_16 = '/media/veracrypt2/gloria/H36M/h36m_16b'

# Corresponding action names to their IDs
#corr = {'action_name': ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether'], 'action_id': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]}         # This has numbers based on GT annotations

# This one was modified to match the naming convention in the subject TOF folders
corr = {'action_name': ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'TakingPhoto', 'Photo', 'Waiting', 'Walking', 'WalkingDog', 'WalkDog', 'WalkTogether'], 'action_id': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 13, 14, 15, 15, 16]}


# The following intrinsic camera params were taken from: https://github.com/geopavlakos/c2f-vol-demo/issues/6 (the person that calculated them)
        # Modified so it's all just one big dictionary
S1 = np.array([[248, 0,  100.0], [0, 248, 75], [0, 0, 1.0]], dtype=np.float32)
S5 = np.array([[248, 0, 102], [0, 248, 75], [0, 0, 1.0]], dtype=np.float32)
S6 = np.array([[248, 0, 103.5], [0, 248, 76], [0, 0, 1.0]], dtype=np.float32)
S7 = np.array([[248, 0,  102.5], [0, 248, 75], [0, 0, 1.0]], dtype=np.float32)
S8 = np.array([[248, 0,  105.8], [0, 248, 79], [0, 0, 1.0]], dtype=np.float32)
S9 = np.array([[248, 0,  107.5], [0, 248, 78.5], [0, 0, 1.0]], dtype=np.float32)
S11 = np.array([[248, 0,  108], [0, 248, 80], [0, 0, 1.0]], dtype=np.float32)

cam_ints = {'S1' : S1, 'S5' : S5, 'S6' : S6, 'S7' : S7, 'S8' : S8, 'S9' : S9, 'S11' : S11}

# Setting it as such since we are assuming that the TOF extrinsic matrix is the same as the color one
rt = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)


# Let's organize this properly now
    # camera 2 corresponds to camera "55011271" in the following file, from https://github.com/karfly/human36m-camera-parameters/blob/master/camera-parameters.json
distcoeffs_color = np.array([-0.194213629607385, 0.240408539138292, -0.0027408943961907, -0.001619026613787, 0.00681997559022603], dtype=np.float32)

#subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
subjects = ['S7', 'S8', 'S9', 'S11']

for subject in subjects:
     print(' ')
     print(' ')
     print('LOOKING AT SUBJECT:                  ', subject)

# This was figured out from inspecting the subject#_data.json files. Not all of them have the same output size :( (see page 4 in wooden notebook)
     if subject == 'S7':
         out_size = (1000, 1002)            # (width, height)
     else:
         out_size = (1000, 1000)
     
     camera_path = '/media/veracrypt2/gloria/H36M/annotations/Human36M_subject' + subject[1:] + '_camera.json'
     
     with open(camera_path, 'r') as CAM:
         cameras = json.load(CAM)    # only use this for the extrinsics (CAMERA 2)
     
     Kd = cam_ints[subject]         # intrinsic matrix of the depth camera
     
     cf = cameras['2']['f']
     cc = cameras['2']['c']
     Kc = np.array([[cf[0], 0, cc[0]], 
                    [0, cf[1], cc[1]],
                    [0, 0, 1]], dtype=np.float32)

     tof_root = '/media/veracrypt2/gloria/H36M/' + subject + '/TOF'
     trials = sorted(os.listdir(tof_root))

     print('NUMBER OF TRIALS:                    ', len(trials))
     for trial in trials:
         print('        ORIGINAL TRIAL NAME:        ', trial)
         tof_p = tof_root + '/' + trial

         # Getting rid of dots and spaces if there's any
         trial = trial.split('.')[0]
         trial = trial.split(' ')[0]

         # Checking if there any numbers present
         digit = '0'
         for chr in trial:
             if chr.isdigit():
                 digit = chr
         
         trial = trial.split(digit)[0]
         print('        CLEANED TRIAL NAME:         ', trial)

         trial_idx = corr['action_name'].index(trial)
         number = str(corr['action_id'][trial_idx]).zfill(2)

         imgs_root = '/media/veracrypt2/gloria/H36M/images/images/'
         # two possibilities for images, either subact 1 or 2. Only way to know how is to check lengths
         possible = ['s_' + subject[1:].zfill(2) + '_act_' + number + '_subact_01_ca_02', 's_' + subject[1:].zfill(2) + '_act_' + number + '_subact_02_ca_02']        # we only want camera 2 imgs

         l1 = len(os.listdir(imgs_root + possible[0]))
         l2 = len(os.listdir(imgs_root + possible[1]))
         if l1 == l2:
             print('WE HAVE A PROBLEM: subaction lengths are the same!!')

         tof_file = cdflib.CDF(tof_p)
         tof_index = tof_file.varget('Index')               # shape = 1, x                  where x = number of rgb frames for that trial
         tof_range = tof_file.varget('RangeFrames')         # shape = 1, 144, 176, y        where y = number of TOF frames for that trial

         if len(tof_index[0]) == l1:
             useme_root = imgs_root + possible[0]
             name = possible[0]
         elif len(tof_index[0]) == l2:
             useme_root = imgs_root + possible[1]
             name = possible[1]
         else:
             print('ERROR: tof .cdf index list length is not equal to any of the number of images in the folder!')
         
         print('        CORRESPONDING IMGS FOLDER:  ', name)
         print('LOOPING THROUGH THE TOF DATA')
         
         #for df in tqdm(range(tof_range[3])):
         dp = 'first'
         for rgb_i in tqdm(range(len(tof_index[0]))):
             di = int(tof_index[0][rgb_i])
             if di != dp:
                 depth = tof_range[0,:,:,di-1] * 1000           # di - 1 cause python is 0-indexed
                                                                # multiplying by 1000 to convert it from meters to mm
                 depth = depth.astype(np.uint16)

                 new_name = new_p + '/' + name + '/' + name + '_' + str(int(rgb_i+1)).zfill(6) + '.png'
                 new_depth_name = new_16 + '/' + name + '/' + name + '_' + str(int(rgb_i+1)).zfill(6) + '_DEPTH.png'     # rgb_i + 1 cause its indexed from 0

                 # Checking if the paths exist, creating it if not
                 rgbd_dir = os.path.dirname(new_name)
                 if not os.path.exists(rgbd_dir):
                     os.makedirs(rgbd_dir)
                
                 d_dir = os.path.dirname(new_depth_name)
                 if not os.path.exists(d_dir):
                     os.makedirs(d_dir)

                 # so far the data is of float32 dtype which is what we want for the uint16 individidual files!
                 img_p = useme_root + '/' + name + '_' + str(int(rgb_i+1)).zfill(6) + '.jpg'

                 rgb_pil = Image.open(img_p)
                 rgb_np = np.array(rgb_pil)

##############################################################################################################
# REGISTERING THE UINT16 DEPTH DATA
                 # Register the depth
                 d_np = cv2.rgbd.registerDepth(Kd, Kc, distcoeffs_color, rt, depth, out_size)
                 
                 # Dilating the image to remove noise
                 #kernel = np.ones((3, 3))
                 kernel = np.ones((5, 5))
                 d_np = cv2.dilate(d_np, kernel, iterations=1)

                 # Concatenating both arrays and create the image - NOTE THAT THIS IS ONLY SAVING THE RAW DEPTH DATA
                 d_channel = np.expand_dims(d_np, axis=-1)
                 glo = Image.fromarray(d_channel, mode='I;16')
                 glo.save(new_depth_name)
                 
##############################################################################################################
# NOW REGISTERING THE DEPTH DATA AS A WHOLE PNG IMAGE                        
                 # CODE COPIED OVER FROM CREATE_RGBD_DATA_DANCE5.py in Dite folder
                 # Register the depth
                 d_np = cv2.rgbd.registerDepth(Kd, Kc, distcoeffs_color, rt, depth, out_size)

                 # Dilating the image to remove noise
                 #kernel = np.ones((3, 3), np.uint8)
                 kernel = np.ones((5, 5), np.uint8)
                 d_np = cv2.dilate(d_np, kernel, iterations=1)

                 # Normalizing depth values
                 d_np = (d_np / 5500 * 255)

                 # Clipping anything over 255
                 d_np = np.clip(d_np, 0, 255)
                 # Converting to integers
                 d_np = d_np.astype(np.uint8)

                 # Concatenating both arrays and create the image
                 d_channel = np.expand_dims(d_np, axis=-1)
                 rgbd = np.concatenate((rgb_np, d_channel), axis=-1)
                 image_pil = Image.fromarray(rgbd, 'RGBA')
                 image_pil.save(new_name)
             
             
             # Setting previous = current so we can check the next one and really just keep the ones that have depth
             dp = di
          