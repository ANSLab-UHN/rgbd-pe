
import cv2
import json
import numpy as np
from tqdm import tqdm
from PIL import Image

img_root_p =  input('Enter root folder where images are located:              ')
important_p = input('Enter path of the dictionary with aligned data info:     ')
new_p =       input('Enter folder path where rgbd images will be saved:       ')

ogdata_p = '/media/veracrypt1/gloria/Dite-HRNet/data/cmu/git/OGDATA/'
out_size = (1920,1080)

# Open the dictionary with all time-aligned indices
with open(important_p, 'r') as GT:
    try:
        info = json.load(GT)
    except json.decoder.JSONDecodeError:
        print('The file at ', important_p, 'does not contain valid data')

# Looping through the trial names             
for trial in info.keys():
    print(' ')
    print('LOOKING AT TRIAL:        ', trial)

# Looping through the cameras
    for j in range(10):
        cam_num = str(j + 1)
        node = 'KINECTNODE' + str(cam_num)
        print('Camera:                  ', node)

        rgb_info = info[trial]['NEW_KNAME'][cam_num]
        d_info = info[trial]['KDEPTH'][cam_num]

        # Opening up the calibration table
        calib_p = ogdata_p + trial + '/kcalibration_' + trial + '.json'
        #print('Calibration table file path:          ', calib_p)

        with open(calib_p, 'r') as TABLE:
            try:
                calib = json.load(TABLE)
            except json.decoder.JSONDecodeError:
                print('The file at ', calib_p, 'does not contain valid data')
        
        cam_calib = calib['sensors'][j]

        distcoeffs_color = np.array(cam_calib['distCoeffs_color'], np.float32)

        T1 = np.array(cam_calib['M_color'], np.float32)
        T2 = np.array(cam_calib['M_depth'], np.float32)
        T1_inv = np.linalg.inv(T1)
        rt = np.dot(T1_inv, T2)         # rigid transform matrix

        Kc = np.array(cam_calib['K_color'], np.float32)
        Kd = np.array(cam_calib['K_depth'], np.float32)
        K_color = np.hstack((Kc, np.zeros((3, 1))))
        K_depth = np.hstack((Kd, np.zeros((3, 1))))

        P_color = np.dot(K_color, T1)
        P_depth = np.dot(K_depth, T2)
        #print('Camera-specific calibration parameters loaded')

        # Opening the depth file
        d_p = ogdata_p + trial + '/kinect_shared_depth/' + node + '/depthdata.dat'
        with open(d_p, 'rb') as f:

# Making sure that the lists have the same lengths
            if len(rgb_info) == len(d_info):

                # Looping through each value
                for i in tqdm(range(len(rgb_info))):

    # Making sure that the values are valid and not supposed to be skipped
                    if rgb_info[i] not in ['NEWskip', 'skip'] and d_info[i] not in ['no', 'dskip']:
                        
                        # Open the image and get the np array from it
                        img_p = img_root_p + '/' + rgb_info[i]
                        #print('Image old path:          ', img_p)
                        rgb_pil = Image.open(img_p)
                        rgb_np = np.array(rgb_pil)

                        # Create a new file name for the depth
                        new_name = new_p + '/' + rgb_info[i][0:-3] + 'png'
                        # Start the depth registration process
                        #print('Depth data path:         ', d_p)
                        idx = d_info[i]

                        # Read depth frame at file index (1-based) idx
                        f.seek(2 * 512 * 424 * (idx - 1), 0)
                        d_img = np.fromfile(f, dtype=np.uint16, count=512*424)  
                        im = d_img.reshape((424, 512))
                        im = np.fliplr(im)

                        # Register the depth
                        d_np = cv2.rgbd.registerDepth(Kd, Kc, distcoeffs_color, rt, im, out_size)

                        # Dilating the image to remove noise
                        kernel = np.ones((3, 3), np.uint8)
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

                    else:
                        print('ERROR: skip messages are not adding up')
                        print(rgb_info[i])
                        print(d_info[i])

            
            else:
                print('ERROR: rgb length != d length')
