# This code was created on march 10 to correct the fact that old alignedDepth dicts (pre newSplit) NEW_KNAME renaming was done based on the newSplit GT files
# Prompts user for a limit, checks that ids are good based on the start of the file name before the first underscore
import cv2
import json
import numpy as np
from tqdm import tqdm
from PIL import Image

img_root_p =  input('Enter root folder where images are located:              ')
important_p = input('Enter path of the dictionary with aligned data info:     ')
new_p =       input('Enter folder path where rgbd images will be saved:       ')

ignore = ['190986', '190987', '190988', '190989', '190990', '190991', '190992', '190993', '190994', '190995', '190996', '190997', '190998', '190999', '191000', '191001', '191002', '191003', '191004', '191005', '191006', '191007', '191008', '191009', '191010', '191011', '191012', '191013', '235601', '235602', '235603', '235604', '235605', '235606', '235607', '235608', '235609', '235610', '235611', '235612', '235613', '235614', '235615', '235616', '235617', '235618', '235619', '235620', '235621', '235622', '235623', '235624', '235625', '235626', '235627', '235628', '235629', '235630', '235631', '235634', '235635', '246756', '246757', '246758', '246759', '246760', '246761', '246762', '246763', '246764', '246765', '246766', '246767', '246768', '246769', '246770', '246771', '246772', '246773', '246774', '246775', '246776', '246777', '246778', '246779', '246780', '246781', '246782', '246783', '246784', '246785', '246786', '246787', '246788', '246789', '257152', '257153', '257155', '257156', '257157', '257158', '257159', '257165', '257911', '257912', '257913', '257914', '257915', '257916', '257917', '257918', '265431', '286969', '286970', '286971', '286972', '286973', '286974', '291370', '291371', '291372', '291373', '291374', '291375', '291376', '291377', '291378', '291379', '291380', '291381']

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
                        
                        if rgb_info[i].split('_', 1)[0] not in ignore:
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

                            # Normalizing depth values (used to be d_np = (d_np / 8000 * 255).astype(np.uint8) -> issues if value > 8000)
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
