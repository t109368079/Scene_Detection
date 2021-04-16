# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 12:21:39 2021

@author: NTUT
"""

import os
import cv2

def load_shotBoundary(boundary_path):
    fp = open(boundary_path,'r').readlines()[0].replace('[','').replace(']','').replace(' ','')
    boundary = [int(each) for each in fp.split(',')]
    return boundary

boundary_dir = 'G:/OVSD_Dataset/annotation/shots'
frame_dir = 'G:/OVSD_Dataset/video_frame'
shot_dir = 'G:/OVSD_Dataset/video'
video_list = ['BoyWhoNeverSlept','CosmosLaundromat','fires_beneath_water','Honey',
              'La_chute_d_une_plume','Meridian_Netflix','Route_66','Seven Dead Men','SitaSingsMovie','StarWreck']

for video_name in video_list:
    boundary_path = os.path.join(boundary_dir,video_name+'.txt')
    boundary = load_shotBoundary(boundary_path)
    
    frame_path = os.path.join(frame_dir,video_name)
    total_frame = len(os.listdir(frame_path))
    
    if total_frame>boundary[-1]+30:
        boundary.append(total_frame-1)  
    else:
        boundary[-1] = total_frame-1
    shot_path = os.path.join(shot_dir,video_name)
    if not os.path.isdir(shot_path):
        os.mkdir(shot_path)
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    
    if os.path.isfile(os.path.join(shot_dir,video_name+'.mp4')):
        videoCap = cv2.VideoCapture(os.path.join(shot_dir,video_name+'.mp4'))
    elif os.path.isfile(os.path.join(shot_dir,video_name+'.avi')):
        videoCap = cv2.VideoCapture(os.path.join(shot_dir,video_name+'.avi'))
    else:
        raise RuntimeError('Video not found!')
    fps = videoCap.get(cv2.CAP_PROP_FPS)
    print('{}, fps={}'.format(video_name,fps))
    
    start_frame = 0
    shot_index = 0
    for end_frame in boundary:
        shot_frame = []
        print('Processing shot {}'.format(shot_index))
        for i in range(start_frame,end_frame+1):
            img = cv2.imread(os.path.join(frame_path,'frame_{}.jpg'.format(i)))
            shot_frame.append(img)
            h,w,_ = img.shape
        size = (w,h)
        shot_name = os.path.join(shot_path,'shot_{}.avi'.format(shot_index))
        out = cv2.VideoWriter(shot_name,fourcc,fps,size)
        for frame in shot_frame:
            out.write(frame)
        out.release()
        start_frame = end_frame+1
        shot_index += 1
        
    

