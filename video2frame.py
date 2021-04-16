# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 11:29:50 2021

@author: NTUT
"""

import os
import cv2 
import numpy as np

bbc_dir = 'C:/Users/NTUT/Google 雲端硬碟/bbc_dataset_video'
tmp = os.listdir(bbc_dir)
bbc_video_list=[]
for each in tmp:
    if each.endswith('.mp4'):
        bbc_video_list.append(each)
bbc_frame_dir = 'G:/bbc_earth_Dataset/video_frame'

video_list = ['02_Mountains','03_Ice_Worlds','04_Great_Plains','05_Jungles','06_Seasonal_Forests','07_Fresh_Water',
                  '08_Ocean_Deep','09_Shallow_Seas','10_Caves','11_Deserts']

for video_name in video_list:
    print('Processing {}'.format(video_name))
    if not os.path.isdir(os.path.join(bbc_frame_dir,video_name.replace('.mp4',''))):
        frame_dir = os.path.join(bbc_frame_dir,video_name.replace('.mp4',''))
        os.mkdir(frame_dir)
        
    videoCap = cv2.VideoCapture(os.path.join(bbc_dir,video_name+'.mp4'))
    count = 0
    success, img = videoCap.read()
    while success:
        filename = os.path.join(frame_dir,'frame{}.jpg'.format(count))
        cv2.imwrite(filename, img)
        success, img = videoCap.read()
        count += 1