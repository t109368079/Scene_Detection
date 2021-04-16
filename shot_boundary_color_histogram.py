# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 11:54:35 2021

@author: NTUT
"""

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def histDiff(ha,hb,img_size):
    a = np.reshape(ha,-1)
    b = np.reshape(hb,-1)
    diff = 0
    for i in range(len(a)):
        tmp = abs(int(a[i])-int(b[i]))
        diff += tmp
    return diff/(img_size[0]*img_size[1]*img_size[2])

def get_boundary(diff):
    boundary = []
    prev_boundary=0
    for i in range(len(diff)):
        if diff[i] >= 0.5 and i>(prev_boundary+30):
            prev_boundary = i
            boundary.append(i)
    return boundary

def load_gt(gt_path):
    fp = open(gt_path).readlines()
    _, end = fp[0].split('\t')
    end = int(end)
    gt_boundary = []
    for i in range(1,len(fp)):
        start, next_end = fp[i].split('\t')
        start = int(start)
        next_end = int(next_end)
        gt_boundary.append([i for i in range(end,start)])
        end = next_end
    return gt_boundary
    
def evaluate(pred,gt):
    tp = 0
    for gt_boundary in gt:
        start = gt_boundary[0]
        end = gt_boundary[-1]
        for pred_boundary in pred:
            if pred_boundary<start:
                continue
            elif pred_boundary in gt_boundary:
                tp += 1
                break
            elif pred_boundary>end:
                break
    precision = tp/len(pred)
    recall = tp/len(gt)
    fscore = 2*precision*recall/(precision+recall)
    print('fscore is {}'.format(fscore))
    return fscore
        


video_frame_dir = 'G:/OVSD_Dataset/video_frame'
video_list = ['Seven Dead Men','SITA_SINGS_MOVIE_ONLY','Star Wreck- In the Pirkinning']
fscore_list = []
for video_name in video_list:
    frame_dir = os.path.join(video_frame_dir,video_name)
    
    total_frame = len(os.listdir(frame_dir))
    
    diff = np.zeros(total_frame-1)
    
    framei = cv2.imread(os.path.join(frame_dir,'frame_{}.jpg'.format(0)))
    img_size = framei.shape
    histi = np.zeros((256,3))
    for j in range(3):
        histi[:,j] = cv2.calcHist([framei], [j], None, [256], [0, 256]).T
        
    for i in range(1,total_frame):
        if (i%10000 == 0):
            print("Processing frame_{}".format(i))        
        frameNext = cv2.imread(os.path.join(frame_dir,'frame_{}.jpg'.format(i)))
        histNext = np.zeros((256,3))
        
        for j in range(3):
            histNext[:,j] = cv2.calcHist([frameNext], [j], None, [256], [0,256]).T
        diff[i-1] = histDiff(histi,histNext,img_size)
        histi = histNext
    
    boundary = get_boundary(diff)
    save_name = os.path.join('G:/OVSD_Dataset/annotation/shots/',video_name+'.txt')
    with open(save_name,'w') as fw:
        fw.write(str(boundary))
    print('Finish {}'.format(video_name))
    # gt_dir = 'C:/Users/NTUT/Google 雲端硬碟/annotations/shots'
    # gt_path = os.path.join(gt_dir,video_name+'.txt')
    # gt_boundary = load_gt(gt_path)
    # fscore = evaluate(boundary, gt_boundary)
    # fscore_list.append(fscore)
