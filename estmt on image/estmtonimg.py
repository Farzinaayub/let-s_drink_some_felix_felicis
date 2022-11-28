import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
from IPython.display import HTML
import posemodulecopy as pm
import csv

detector = pm.poseDetector()
def rescale_frame(frame, percent):
        width = int(frame.shape[1] * percent/ 100)
        height = int(frame.shape[0] * percent/ 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=0)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils 
newlist=[]
head = []
head.append(['Asana','nose_x','nose_y','left_eye_inner_x','left_eye_inner_y','left_eye_x','left_eye_y',
            'left_eye_outer_x','left_eye_outer_y','right_eye_inner_x','right_eye_inner_y','right_eye_x','right_eye_y',
            'right_eye_outer_x','right_eye_outer_y','left_ear_x','left_ear_y','right_ear_x','right_ear_y','mouth_left_x','mouth_left_y',
            'mouth_right_x','mouth_right_y','left_shoulder_x','left_shoulder_y','right_shoulder_x','right_shoulder_y',
            'left_elbow_x','left_elbow_y','right_elbow_x','right_elbow_y','left_wrist_x','left_wrist_y','right_wrist_x','right_wrist_y',
            'left_pinky_x','left_pinky_y','right_pinky_x','right_pinky_y','left_index_x','left_index_y','right_index_x','right_index_y',
            'left_thumb_x','left_thumb_y','right_thumb_x','right_thumb_y','left_hip_x','left_hip_y','right_hip_x','right_hip_y','left_knee_x','left_knee_y',
            'right_knee_x','right_knee_y','left_ankle_x','left_ankle_y','right_ankle_x','right_ankle_y','left_heel_x','left_heel_y',
            'right_heel_x','right_heel_y','left_foot_index_x','left_foot_index_y','right_foot_index_x','right_foot_index_y'])

#Input_Image
flag=0

csvlst =[]

sample_img  = cv2.imread(r'model\test_yoga\sidhasana\17.jpg')
sample_img = detector.findPose(sample_img)
lmlist = detector.findPostition(sample_img)
for i in range(0,len(lmlist)):
    newlist = newlist+lmlist[i]

print(newlist)
if len(newlist)!=0:
    file = open('tree.csv', 'a', newline ='')
            # writing the data into the file
with file as obj:
    write = csv.writer(obj)
    #write.writerow(head)
        #flag=flag+1
    write.writerow(newlist)
    obj.close()

#Check if any landmarks are found.
if detector.results.pose_landmarks:
    
    #Draw Pose landmarks on the sample image.
    mp_drawing.draw_landmarks(image=sample_img, landmark_list=detector.results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)
       
    #Specify a size of the figure.
    fig = plt.figure(figsize = [10, 10])
    sample_img = rescale_frame(sample_img, percent=100)
    #Display the output image with the landmarks drawn, also convert BGR to RGB for display. 
    plt.title("Output");plt.axis('off');plt.imshow(sample_img[:,:,::-1]);plt.show()