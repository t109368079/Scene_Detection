# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 15:56:09 2021

@author: NTUT
"""

import os
import torch
import numpy as np
from sklearn.cluster import KMeans

def createScene(src):
    shotInScene = []
    for i in range(len(src)-1):
        if i == 0:
            start_shot = 0
            end_shot = src[i+1]
        else:
            start_shot = src[i]
            end_shot = src[i+1]
        scene = [shot for shot in range(start_shot,end_shot)]
        shotInScene.append(scene)
    shotInScene[-1].append(src[-1])
    return shotInScene

def convert_gt(gt_path,new_path):
    """
    To get ketShot, gt_file must be pair (start_shot, end_shot)
    It convert scquence of boundary to (start_shot, end_shot)

    Parameters
    ----------
    gt_path : str
        path to GT.

    """
    fp = open(gt_path,'r').readlines()
    if os.path.isfile(new_path):        
        if(input(Warning('{} exist, press Enter to overwrite, anykey to stop: ')) == ''):
            fw = open(new_path,'w')
        else:
            raise KeyboardInterrupt()
    else:
        fw = open(new_path,'w')
    boundary = fp[0].split(',')
    start = 0
    for b in boundary:
        end = int(b)
        fw.write('{}\t{}\n'.format(start,end))
        start = end
    fw.close()
    fp.close()
    print("Generate {}".format(new_path))
    
def createGt(keyshots, scenes,gt_path):
    assert len(keyshots) == len(scenes)
    fw = open(gt_path,'w')
    for i in range(len(keyshots)-1):
        scene = scenes[i]
        scene_start = scene[0]
        keyShot = scene_start+keyShots[i]
        fw.write('{},'.format(keyShot))
    scene = scenes[-1]
    scene_start = scene[0]
    keyShot = scene_start+keyShots[-1]
    fw.write('{}'.format(keyShot))
    fw.close()
    print("Generate {}".format(gt_path))
        

def load_scene(gt_path):
    fp = open(gt_path,'r').readlines()
    scenes = []
    for each in fp:
        start, end = each.split('\t')
        start = int(start)
        end = int(end)
        scene = [k for k in range(start,end+1)]
        scenes.append(scene)
    return scenes

def find_center(center,scene_features):
    shortest = np.inf
    for i in range(len(scene_features)):
        shot_feature = scene_features[i]
        dist = np.linalg.norm(shot_feature-center)
        if dist<shortest:
            keyShot = i
            shortest = dist
    return keyShot

if __name__ == '__main__':
    gt_dir = 'G:/OVSD_Dataset/label'
    feature_dir = 'G:/OVSD_Dataset/parse_data/'
    video_list = ['BigBuckBunny','BoyWhoNeverSlept','CosmosLaundromat','fires_beneath_water',
                  'Honey','La_chute_d_une_plume','Meridian_Netflix','Route_66','Seven Dead Men','StarWreck']

   
    
    for video_name in video_list:
        gt_path = os.path.join(gt_dir,video_name+'_shotIndex.txt')
        scenes = load_scene(gt_path)
        total_shot = len(os.listdir(os.path.join(feature_dir,video_name)))
        kmean = KMeans(n_clusters=1)
        keyShots = []
        for scene in scenes:
            scene_features = np.empty((len(scene),4096))
            for i in range(len(scene)):
                shot = scene[i]
                if(shot == total_shot):
                    scene_features = np.delete(scene_features,i,0)
                    break
                tmp = torch.load(os.path.join(feature_dir,video_name,'shot_{}.pt'.format(shot)))
                tmp = tmp.cpu()
                tmp = tmp.detach().numpy()
                scene_features[i] = tmp
            kmean.fit(scene_features)
            center = kmean.cluster_centers_
            keyShots.append(find_center(center, scene_features))
        keyShot_path = os.path.join(gt_dir,video_name+'_keyshot.txt')
        createGt(keyShots,scenes,keyShot_path)

            
            
