# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 09:39:10 2021

@author: NTUT
"""

import os
import numpy as np

def load_shot_boundary(path):
    fp = open(path,'r').readlines()[0].replace('[','').replace(']','').replace(' ','')
    boundary = [int(each) for each in fp.split(',')]
    return boundary

def load_scene(path):
    fp = open(path,'r').readlines()
    scenes = []
    for each in fp:
        start, end = each.split('\t')
        start = int(start)
        end = int(end)
        scenes.append((start,end))
    return scenes

def find_shot(index,shot_boundary,start=None):
    if index > shot_boundary[-1]:
        return len(shot_boundary)
    if start is not None:
        if start == 0:
            start_frame = 0
        else:
            start_frame = shot_boundary[start-1]+1
        for i in range(start,len(shot_boundary)):
            end_frame = shot_boundary[i]+1
            if index in range(start_frame,end_frame):
                return i
            else:
                start_frame = end_frame
    else:
        start_frame = 0
        for i in range(len(shot_boundary)):
            end_frame = shot_boundary[i]+1
            if index in range(start_frame,end_frame):
                return i
            else:
                start_frame = end_frame

def write_scenegt(scene_shot,gt_name,path=None):
    if path is None:
        path = '../'
    gt_path = os.path.join(path,gt_name+'.txt')
    fw = open(gt_path,'w')
    for scene in scene_shot:
        fw.writelines('{}\t{}\n'.format(scene[0],scene[1]))
    fw.close()
            
                    

if __name__ == '__main__':
    ovsd_dir = 'G:/OVSD_Dataset'
    shot_dir = os.path.join(ovsd_dir,'annotation','shots')
    scene_dir = os.path.join(ovsd_dir,'label')
    video_list = ['BigBuckBunny','BoyWhoNeverSlept','CosmosLaundromat','fires_beneath_water',
                  'Honey','La_chute_d_une_plume','Meridian_Netflix','Route_66','Seven Dead Men','SitaSingsMovie','StarWreck']
    for video_name in video_list:
        shot_boundary = load_shot_boundary(os.path.join(shot_dir,video_name+'.txt'))
        scenes = load_scene(os.path.join(scene_dir,video_name+'.txt'))
        scene_shot = []
        nextStart = 0
        for scene in scenes:
            start = scene[0]
            end = scene[1]
            start_shot = find_shot(start, shot_boundary,nextStart)
            end_shot = find_shot(end,shot_boundary,start_shot)
            scene_shot.append((start_shot,end_shot))
            nextStart = end_shot
        write_scenegt(scene_shot, '{}_shotIndex'.format(video_name),path='G:/OVSD_Dataset/label')    
        
        
        
        
    
