# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 14:56:00 2021

@author: NTUT
"""

import os
import numpy as np


video_list = ['01_From_Pole_to_Pole','02_Mountains','03_Ice_Worlds','04_Great_Plains','05_Jungles','06_Seasonal_Forests','07_Fresh_Water',
                  '08_Ocean_Deep','09_Shallow_Seas','10_Caves','11_Deserts']

result = np.empty((5,len(video_list)))

for i in range(5):
    gt_dir = '../annotations/scenes/annotator_'+str(i)
    for j in range(len(video_list)):
        video_name = video_list[j]
        gt = os.path.join(gt_dir,video_name+'.txt')
        f = open(gt,'r').readlines()
        data = f[0].split(',')
        result[i][j] = len(data)
        