# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 10:17:01 2021

@author: NTUT
"""

import os
import math
import time
import random
import numpy as np
from datetime import datetime
from coverage_overflow import fscore_eval

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
    
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class MyTransformer(nn.Module):
    def __init__(self,d_model,nhead,num_layer,windowSize):
        super(MyTransformer,self).__init__()
        ##Model Parameter
        self.d_model = d_model
        self.nhead = nhead
        self.num_lay = num_layer
        #Model Architecture
        self.pe = PositionalEncoding(d_model)
        encoderLayer = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(encoderLayer, num_layer)
        self.layer1 = nn.Linear(d_model,2048)
        self.layer2 = nn.Linear(2048,1024)
        self.layer3 = nn.Linear(1024,windowSize)
        
    def forward(self,shot_feature):
        num_shot = shot_feature.shape[0]
        src = shot_feature.view(num_shot,1,self.d_model)
        src = self.pe(src)
        attention_out = self.encoder(src)
        x = F.relu(self.layer1(attention_out))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        
        return x
    
def train_test_split(dataset,test_size=0.3):
    nvideo = len(dataset)
    ntest = int(test_size*nvideo)
    index = [i for i in range(nvideo)]
    random.shuffle(index)
    training = [dataset[i] for i in index[ntest:]]
    testing = [dataset[i] for i in index[:ntest]]
    return training, testing

def visual_feature(visual_dir):
    listShots = os.listdir(visual_dir)
    nShot = len(listShots)
    feature = torch.empty((nShot,4096))
    for i in range(nShot):
        visual_feature_path = os.path.join(visual_dir,listShots[i])
        tmp = torch.load(visual_feature_path,map_location=torch.device('cpu'))
        feature[i,:] = tmp
    return feature

def load_gt(gt_path):
    tmp = open(gt_path,'r').readlines()
    tmp = [each.replace('\n','').replace('[','').replace(']','') for each in tmp]
    gt = []
    for each in tmp[0].split(','):
        gt.append(int(each))
    gt = torch.tensor(gt)
    return gt

def load_keyShot(video_name):
    tmp = open('../annotations/scenes/annotator_1/{}.txt'.format(video_name)).readlines()
    scene_boundary = [int(each) for each in tmp[0].split(',')]
    tmp = open('../annotations/scenes/annotator_1/{}_keyshot.txt'.format(video_name)).readlines()
    keyShots = [int(each) for each in tmp[0].split(',')]
    key_gt = []
    for i in range(len(keyShots)):
        key = keyShots[i]
        length = scene_boundary[i+1]-scene_boundary[i]
        key_gt += [key for i in range(length)]
    key_gt = torch.tensor(key_gt)
    return key_gt

def clean_gt(gt,start_shot):
    for i in range(len(gt)):
        gt[i] = max(0,gt[i]-start_shot)
        gt[i] = min(29,gt[i])

def fix_pred(gt,start_shot):
    for i in range(len(gt)):
        gt[i] = gt[i]+start_shot

        
def pred_scenes(pred,mask=8):
    """

    Parameters
    ----------
    pred : torch.tensor
        pred are top 5 shot current shot attention to.
    mask : int, optional
        In pred, only care about the shot in range current index-mask to current index+mask The default is 8.

    Returns
    -------
    boundary : list
        return scene boundary represented by shot index. (Not 0 and 1)

    """
    pred_np = pred.detach().numpy()
    total_shot = len(pred_np)
    links = []
    for i in range(total_shot):
        attention_to = pred_np[i]
        lower = max(0,i-mask)
        upper = min(total_shot,i+mask)
        noLink = True
        for each in attention_to:
            if each in range(lower,upper):
                links.append((i,each))
                noLink = False
                break
        if noLink:
            links.append((i,i))    
    scenes = []        
    for link in links:
        start = int(min(link))
        end = int(max(link))
        new_scene = [i for i in range(start,end+1)]
        isNew = True
        for i in range(len(scenes)):
            if start in scenes[i]:
                scenes[i] += [s for s in range(scenes[i][-1]+1,end+1)]
                isNew = False
                break
        if isNew and len(new_scene)>0:
            scenes.append(new_scene)
    del links
        
    boundary = []
    tmp_point = 0
    for pred_scene in scenes:
        if pred_scene[-1]>tmp_point:
            boundary.append(pred_scene[-1])
            tmp_point = pred_scene[-1]
    del scenes
    return np.array(boundary)


def model_save(model,save_name=None,epoch=None,save_path=None):
    if (save_name is None) and (epoch is None):
      save_name = datetime.today().strftime('%Y%m%d')+'_model.pt'
    elif epoch is not None:
      save_name = datetime.today().strftime('%Y%m%d')+'_epoch{}_model.pt'.format(str(epoch))
    if save_path is None:
      save_path = 'G:/model'
    
    save_path = os.path.join(save_path,save_name)
    torch.save(model.state_dict(),save_path)
    print("Saving Model {}".format(save_name))
    
def evaluate_window(model,video_list,mask):
    print('evaluating model')
    fscore = 0
    windowSize = 30
    for i in range(len(video_list)):
        video_name = video_list[i]
        visual_feature_path = os.path.join('../parse_data',video_name)
        feature = visual_feature(visual_feature_path).to(device)
        nbatch = int(feature.shape[0]/windowSize)+1
        pred = torch.tensor([]).to(torch.device('cpu'))
        for j in range(nbatch):
            start = j*windowSize
            end = min(feature.shape[0],(j+1)*windowSize)
            src = feature[start:end]
            att_out = model(src)
            _,tmp = torch.topk(att_out.view(-1,windowSize),5)
            fix_pred(tmp,start)
            tmp = tmp.to(torch.device('cpu'))
            pred = torch.cat((pred,tmp))
        boundary = pred_scenes(pred,mask=mask)
        score = fscore_eval(boundary, video_name)
        fscore += score
    fscore = fscore/len(video_list)
    return fscore        
            
if __name__ == '__main__':
    ground_dir = '../'
    video_list = ['01_From_Pole_to_Pole','02_Mountains','03_Ice_Worlds','04_Great_Plains','05_Jungles','06_Seasonal_Forests','07_Fresh_Water',
                  '08_Ocean_Deep','09_Shallow_Seas','10_Caves','11_Deserts']
    cuda = True
    device = torch.device('cuda' if cuda else 'cpu')
    windowSize = 30
    model = MyTransformer(4096, 4, 6, windowSize).to(device)
    lossfun = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5)
    epochs = 30
    eval_rate = 1
    f_score = 0
    
    for epoch in range(epochs):
        train,test = train_test_split(video_list)
        for video_name in train:
            model.train()
            visual_feature_dir = os.path.join(ground_dir,'parse_data',video_name)
            feature = visual_feature(visual_feature_dir).to(device)
            groundtruth = load_keyShot(video_name).to(device)
            nbatch = int(feature.shape[0]/windowSize)+1
            loss = 0
            for i in range(nbatch):
                start = i*windowSize
                end = min((i+1)*windowSize,feature.shape[0])
                src = feature[start:end]
                gt_window = groundtruth[start:end]
                clean_gt(gt_window,start)
                att_out = model(src)
                
                lossout = lossfun(att_out.view(-1,windowSize),gt_window)
                loss += lossout.item()
                del src
                
            del feature
            lossout.backward()
            optimizer.zero_grad()
            optimizer.step()
            print('{} Epoch {}, loss {}'.format(video_name,epoch,loss))
            
        if epoch % eval_rate == eval_rate-1:
            tmp = evaluate_window(model,video_list,mask=8)
            scheduler.step()
            if tmp >f_score:
                f_score = tmp
                best_model = model
            print('Epoch {}, f_score: {}, best_fscore: {}'.format(epoch,tmp,f_score))
    model_save(model,save_name=datetime.today().strftime('%Y%m%d')+'_window_model.pt')
    