import cv2
import numpy as np

# Load the original image
imgs = []
image_path = '/media/veracrypt2/gloria/Participant_data/final_rgbd_imgs/U03/20300291_U03_L01_00291.png'

def crop_string_until_period(input_string):
    # Find the index of the first period in the string
    period_index = input_string.find('.')

    # If a period is found, return the substring until the period
    if period_index != -1:
        cropped_string = input_string[:period_index]
        return cropped_string
    else:
        return input_string  # Return the original string if no period is found

output_filename = '___U03_MHP_rgb_pred.jpg'

print(image_path)
print(output_filename)

image = cv2.imread(image_path)

k = [[687.69519285, 468.15786838,0],
 [883.59655488, 440.01552765,0],
 [883.60607589, 440.01587065,0],
 [883.60108568, 440.01413171,0],
 [883.59985   , 440.01367704,0],
 [711.47693037, 447.76038138,0],
 [652.0449224 , 436.29559257,0],
 [717.22485237, 440.75183539,0],
 [564.68860082, 414.81857273,0],
 [728.8950817 , 430.21414251,0],
 [530.03478542, 402.24929873,0],
 [883.59118445, 440.01092505,0],
 [883.60121241, 440.01465021,0],
 [883.59764797, 440.01644498,0],
 [883.60376297, 440.01312664,0],
 [883.60021437, 440.0178808 ,0],
 [883.60083221, 440.0182238 ,0],]

keypoints = []

for i in range(len(k)):
    for j in range(len(k[0])):
        keypoints.append(k[i][j])

# Reshape keypoints into an array of shape (num_keypoints, 3)
keypoints = np.array(keypoints).reshape(-1, 3)

lines = [(5,6), (6,8), (8,10), (5,7), (7,9), (5,11), (6,12), (11,13), (13,15), (12,14), (14,16), (11,12)]
i = 0

for line in lines:
        start_keypoint = keypoints[line[0]]
        end_keypoint = keypoints[line[1]]
        cv2.line(image, (int(start_keypoint[0]), int(start_keypoint[1])),
                 (int(end_keypoint[0]), int(end_keypoint[1])), (0, 0, 255), 2)
        
for x, y, score in keypoints:
    cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.putText(image, str(i+1), (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    i += 1

cv2.imwrite(output_filename, image)

# Display the image with keypoints
#cv2.imshow('Image', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
