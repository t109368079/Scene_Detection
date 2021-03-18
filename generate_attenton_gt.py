# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 10:05:27 2021

@author: NTUT
"""

import os

def next_shot_gt():
    gt_dir = '../annotations/scenes/annotator_0/'
    video_list = ['07_Fresh_Water']
    
    for video_name in video_list:
        f = open(os.path.join(gt_dir,video_name+'.txt'),'r').readlines()
        gt = [int(each) for each in f[0].split(',')]
        
        gt_new = []
        prev_boundary=0
        for boundary in gt:
            for i in range(prev_boundary+1,boundary):
                gt_new.append(i+1)
            gt_new.append(boundary)
            prev_boundary= boundary
        del gt_new[-1]
        
        fwrite = open(os.path.join(gt_dir,video_name+'_attention.txt'),'w')    
        for i in range(len(gt_new)-1):
            fwrite.write(str(gt_new[i])+',')
        fwrite.write(str(gt_new[-1]))
        fwrite.close()
    
def middle_shot_gt():
    gt_dir = '../annotations/scenes/annotator_1'
    video_list = ['01_From_Pole_to_Pole','02_Mountains','03_Ice_Worlds','04_Great_Plains','05_Jungles','06_Seasonal_Forests','07_Fresh_Water',
                  '08_Ocean_Deep','09_Shallow_Seas','10_Caves','11_Deserts']
    
    for video_name in video_list:
        f = open(os.path.join(gt_dir,video_name+'.txt'),'r').readlines()
        gt = [int(each) for each in f[0].split(',')]
        
        gt_new = []
        prev_boundary=0
        for boundary in gt:
            middle_shot = int((prev_boundary+boundary)/2)
            if middle_shot <=prev_boundary:
                middle_shot = boundary
            for i in range(prev_boundary+1,boundary):
                gt_new.append(middle_shot)
            gt_new.append(middle_shot)
            prev_boundary= boundary
        del gt_new[-1]
        
        fwrite = open(os.path.join(gt_dir,video_name+'_middle_shot_attention.txt'),'w')    
        for i in range(len(gt_new)-1):
            fwrite.write(str(gt_new[i])+',')
        fwrite.write(str(gt_new[-1]))
        fwrite.close()
        print("Generating middle shot attention GT {}".format(video_name))
            

if __name__ == '__main__':
    # next_shot_gt()
    middle_shot_gt()
    