# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 16:26:31 2021

@author: NTUT
"""

import os

def load_gt(gt_path):
    fp = open(gt_path,'r').readlines()
    boundary = [int(each) for each in fp[0].split(',')]
    return boundary

def generate_bgt(total_frame,boundary,gt_path):
    fw = open(gt_path,'w')
    bgt = [0 for i in range(total_frame)]
    for b in boundary:
        try:
            bgt[b] = 1
        except Exception:
            pass
    for i in range(total_frame-1):
        fw.write('{},'.format(bgt[i]))
    fw.write('{}'.format(bgt[-1]))
    return bgt

def generate_shotgt(gt_path,new_path,total_shot):
    fp = open(gt_path,'r').readlines()
    fw = open(new_path,'w')
    for i in range(len(fp)):
        scene = fp[i]
        start, _ = scene.split('\t')
        fw.write('{},'.format(int(start)))
    fw.write('{}'.format(total_shot))
    fw.close()

if __name__ == '__main__':
    gt_dir = 'G:/OVSD_Dataset/label'
    video_list = ['BigBuckBunny','BoyWhoNeverSlept','CosmosLaundromat','fires_beneath_water',
                  'Honey','La_chute_d_une_plume','Meridian_Netflix','Route_66','Seven Dead Men','StarWreck']
    for video_name in video_list:
        total_shot = len(os.listdir(os.path.join('G:/OVSD_Dataset/parse_data',video_name)))
        gt_path = os.path.join(gt_dir,video_name+'_shotIndex.txt')
        shotgt_path = os.path.join(gt_dir,video_name+'_shot.txt')
        generate_shotgt(gt_path, shotgt_path,total_shot)        
        boundary = load_gt(shotgt_path)
        bgt_path = os.path.join(gt_dir,video_name+'_BGT.txt')
        bgt = generate_bgt(total_shot, boundary, bgt_path)

        
    