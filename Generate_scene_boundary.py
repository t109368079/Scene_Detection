# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 09:44:44 2021

@author: NTUT
"""

import os
import torch
import numpy as np
from datetime import datetime
from coverage_overflow import fscore_eval
from scene_detection_TransformerEncoder import MyTransformer, pred_scenes, representShot

def visual_feature(visual_dir):
    listShots = os.listdir(visual_dir)
    nShot = len(listShots)
    feature = torch.empty((nShot,4096))
    for i in range(nShot):
        visual_feature_path = os.path.join(visual_dir,listShots[i])
        tmp = torch.load(visual_feature_path,map_location=torch.device('cpu'))
        feature[i,:] = tmp
    return feature

def load_model(path,textual=False):
    if textual:
        model = MyTransformer(4396,4,6)
    else:
        model = MyTransformer(4096,4,6)
    model.load_state_dict(torch.load(path))
    return model

def clean_boundary(nshot,boundary):
    filter_arr = boundary<nshots
    boundary = boundary[filter_arr]
    if (nshot-1) not in boundary:
        boundary = np.append(boundary,nshot-1)
    return boundary

def write_boundary(boundary,video_name,printed=True):
    save_dir = 'G:/boundary_result/{}'.format(datetime.today().strftime('%Y%m%d'))
    if not (os.path.isdir(save_dir)):
        os.mkdir(save_dir)
    save_name = os.path.join(save_dir,video_name+'_boundary.txt')
    with open(save_name,'w') as fp:
        for i in range(len(boundary)-1):
            item = boundary[i]
            fp.write('{},'.format(item))
        fp.write('{}'.format(boundary[-1]))
    if printed:
        print('Finish writing {}'.format(video_name))
        

        
    
if __name__ == '__main__':
    text = True
    video_list = ['01_From_Pole_to_Pole','02_Mountains','03_Ice_Worlds','04_Great_Plains','05_Jungles','06_Seasonal_Forests',
                  '07_Fresh_Water','08_Ocean_Deep','09_Shallow_Seas','10_Caves','11_Deserts']
    model = load_model('G:/model/20210319_model.pt',textual=text)
    for video_name in video_list:
        visual_feature_dir = os.path.join('../parse_data/',video_name)
        nshots = len(os.listdir(visual_feature_dir))
        if text:
            transcript_path = '../transcript/'
            transcript = open(os.path.join(transcript_path,video_name+'_doc2vec.txt'),'r').readlines()
            transcript = [each.replace('/n','').replace('[','').replace(']','') for each in transcript]
            temp=[]
            for eachShot in transcript:
              temp.append([float(each) for each in eachShot.split(',')])
            transcript = torch.tensor(temp)
            feature = representShot(visual_feature_dir, transcript)
        else:
            feature = visual_feature(visual_dir=visual_feature_dir)
        att_out = model(feature)
        _,pred = torch.topk(att_out.view(-1,nshots),5)
        boundary = pred_scenes(pred)
        boundary = clean_boundary(nshots,boundary)
        score = fscore_eval(boundary, video_name)
        write_boundary(boundary, video_name,printed=False)
        
        
        
        
        