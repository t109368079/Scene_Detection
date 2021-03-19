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

def load_gt(gt_path):
    tmp = open(gt_path,'r').readlines()
    tmp = [each.replace('\n','').replace('[','').replace(']','') for each in tmp]
    gt = []
    for each in tmp[0].split(','):
        gt.append(int(each))
    return gt

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
    gt_dir = '../annotations/scenes/annotator_1/'
    feature_dir = '../parse_data/'
    video_list = ['01_From_Pole_to_Pole','02_Mountains','03_Ice_Worlds','04_Great_Plains','05_Jungles','06_Seasonal_Forests','07_Fresh_Water',
                  '08_Ocean_Deep','09_Shallow_Seas','10_Caves','11_Deserts']
    for video_name in video_list:
        gt_path = os.path.join(gt_dir,video_name+'.txt')
        gt = load_gt(gt_path)
        scenes = createScene(gt)
        kmean = KMeans(n_clusters=1)
        keyShots = []
        for scene in scenes:
            scene_features = np.empty((len(scene),4096))
            for i in range(len(scene)):
                shot = scene[i]
                tmp = torch.load(os.path.join(feature_dir,video_name,'shot_{}.pt'.format(shot)))
                tmp = tmp.detach().numpy()
                scene_features[i] = tmp
            kmean.fit(scene_features)
            center = kmean.cluster_centers_
            keyShots.append(find_center(center, scene_features))
        break
            
            
