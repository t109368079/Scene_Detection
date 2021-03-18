# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 11:50:38 2020

@author: garyy
"""

import os
import torch 
import numpy as np

feature_dir = 'C:/Users/garyy/Google 雲端硬碟/parse_data/'

video_folder_list = os.listdir(feature_dir)

for video in video_folder_list:
    if video.endswith('101'):
        frames = os.listdir(feature_dir+video)
        frames_feature = np.zeros((2048,len(frames)))
        i = 0
        for frame in frames:
            frame_pt = torch.load(feature_dir+video+'/'+frame)
            frame_np = frame_pt.detach().numpy()
            frames_feature[:,i] = frame_np
            i = i+1
            if(i%1000)==0:
                print(video+' frame no.'+str(i))            
        frames_feature_pt = torch.from_numpy(frames_feature)
        torch.save(frames_feature_pt,'../parse_data/'+video+'.pt')
        print('Done video: ',end='')
        print(video)
        del frames_feature
        del frames_feature_pt
    else:
        continue