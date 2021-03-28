# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 09:11:55 2021

This model only use encoder part of transformer
Inputs are C3D feature for each shot
Outputs are the probability of one shot related to another shot
@author: NTUT
"""

import time
import random
import os
import csv
import math
import random
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from coverage_overflow import load_bgt, coverOverflow, fscore_eval

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
    def __init__(self,d_model,nhead,num_layer):
        super(MyTransformer,self).__init__()
        #Model Parameter
        self.d_model = d_model
        self.nhead = nhead
        self.num_layer = num_layer
        #Model Architecture
        self.pe = PositionalEncoding(d_model)
        encoderLayer = nn.TransformerEncoderLayer(self.d_model, self.nhead)
        self.encoder = nn.TransformerEncoder(encoderLayer,self.num_layer)
        self.layer1 = nn.Linear(d_model,2048)
        self.layer2 = nn.Linear(2048,1024)
        self.layer3 = nn.Linear(1024,540)
    
    def forward(self,shot_feature):
        num_shot = shot_feature.shape[0]
        src = shot_feature.view(num_shot,1,self.d_model)
        # src = self.pe(src)
        attention_out = self.encoder(src)
        
        x = F.relu(self.layer1(attention_out))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        
        return x

def visual_feature(visual_dir):
    listShots = os.listdir(visual_dir)
    nShot = len(listShots)
    feature = torch.empty((nShot,4096))
    for i in range(nShot):
        visual_feature_path = os.path.join(visual_dir,listShots[i])
        tmp = torch.load(visual_feature_path,map_location=torch.device('cpu'))
        feature[i,:] = tmp
    return feature
    
def representShot(visual_dir,textual):
    listShots = os.listdir(visual_dir)
    nShot = len(listShots)
    result = torch.empty((nShot,4396))
    for i in range(nShot):
      visual_feature_path = os.path.join(visual_dir,listShots[i])
      if not os.path.isfile(visual_feature_path):
        raise RuntimeError("{} not exist, check file location".format(visual_feature_path))
      else:
        # print("Loading {}".format(visual_feature_path.replace('.pt','')))
        visual = torch.load(visual_feature_path,map_location=torch.device('cpu'))
        text = textual[i]
        shot_feature = torch.cat((visual.view(1,-1),text.view(1,-1)),1)
        result[i,:] = shot_feature
    # print("Load finish, result shape: {}".format(result.shape))
    return result

def check_file(dir_list,ground_dir=None):
    for sub_dir in dir_list:
      print("Checking folder {}".format(sub_dir))
      if ground_dir is not None:
        folder = os.path.join(ground_dir,sub_dir)
      file = os.listdir(folder)
      if len(file)==0 or not os.path.isdir(folder):
        raise RuntimeError('Invaild folder exist, check the path or file in {}'.format(folder))
        return False
      else:
        continue
    print("Checking folder list Done...")
    return True


def train_test_split(dataset,test_size=0.3):
    nvideo = len(dataset)
    ntest = int(test_size*nvideo)
    index = [i for i in range(nvideo)]
    random.shuffle(index)
    training = [dataset[i] for i in index[ntest:]]
    testing = [dataset[i] for i in index[:ntest]]
    return training, testing


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
    
def acc_fun(predList,gt):
    predList = predList.detach().numpy()
    gt = gt.detach().numpy()
    shots = len(gt)
    count = 0
    for ans, pred in zip(gt,predList):
        if ans in pred:
            count = count +1
    acc = count/shots
    return acc

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
    
def scoring(coverage,overflow,printed=True,string=None):
    fscore = []
    for c, o in zip(coverage,overflow):
        if o == 1:
            fscore.append(0)
        else:
            denominator = 1/c+1/(1-o)
            fscore.append(2/denominator)
    fscore = np.array(fscore)
    if printed:
        if string is None:
            raise ValueError("Attribute String are required for print.")
        else:
            print('{}\tF_Score is: {}'.format(string,np.mean(fscore)))
    return np.mean(fscore)    

def train_next_shot(saved=False):
    ground_dir = '../'
    video_list = ['01_From_Pole_to_Pole','02_Mountains','03_Ice_Worlds','04_Great_Plains','05_Jungles','06_Seasonal_Forests','07_Fresh_Water',
                  '08_Ocean_Deep','09_Shallow_Seas','10_Caves','11_Deserts']
    
    transcript_path = os.path.join(ground_dir,'transcript')
    gt_path = os.path.join(ground_dir,'annotations/scenes/annotator_0/')
    cuda = False
    check_file(video_list,ground_dir+'bbc_dataset_video')
    device = torch.device('cuda' if cuda else 'cpu')
    model = MyTransformer(4396,4,6)
    # model = model.cuda()
    lossfun = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5)
    epoches = 4
    eval_rate = 2
    nshots = 540
    f_score = 0
    for epoch in range(epoches):
        loss = 0
        training, testing = train_test_split(video_list)
        print('Epoch :{}...'.format(epoch))
        for video_name in training:
            model.train()
            visual_feature_dir = os.path.join(ground_dir,'parse_data',video_name)
            print("{} Training Start...".format(video_name))
        
            transcript = open(os.path.join(transcript_path,video_name+'_doc2vec.txt'),'r').readlines()
            transcript = [each.replace('/n','').replace('[','').replace(']','') for each in transcript]
            temp=[]
            for eachShot in transcript:
              temp.append([float(each) for each in eachShot.split(',')])
            transcript = torch.tensor(temp)
        
            features = representShot(visual_dir=visual_feature_dir,textual=transcript).to(device)
            groundtruth = load_gt(os.path.join(gt_path,video_name+'_attention.txt')).to(device)
            
            att_out = model(features)
            del features
            
            lossout = lossfun(att_out.view(-1,nshots),groundtruth)
            loss += lossout.item()
            lossout.backward()
            optimizer.zero_grad()
            optimizer.step()
            scheduler.step()
        
        if epoch % eval_rate == eval_rate-1:
            tmp = evaluate(model,video_list)
            if tmp>=f_score:
                f_score = tmp
                best_model = model
            print('Epoch {}, f_score: {}, best_fscore: {}'.format(epoch,tmp,f_score))
            
    if saved:
        model_save(best_model)
    else:
        return best_model

def train_middle_shot(saved=False):
    ground_dir = '../'
    video_list = ['01_From_Pole_to_Pole','02_Mountains','03_Ice_Worlds','04_Great_Plains','05_Jungles','06_Seasonal_Forests','07_Fresh_Water',
                  '08_Ocean_Deep','09_Shallow_Seas','10_Caves','11_Deserts']
    
    transcript_path = os.path.join(ground_dir,'transcript')
    gt_path = os.path.join(ground_dir,'annotations/scenes/annotator_1/')
    cuda = False
    check_file(video_list,ground_dir+'bbc_dataset_video')
    device = torch.device('cuda' if cuda else 'cpu')
    model = MyTransformer(4396,4,6)
    lossfun = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5)
    epoches = 40
    eval_rate = 5
    nshots = 540
    f_score = 0
    for epoch in range(epoches):
        loss = 0
        training, testing = train_test_split(video_list)
        print('Epoch :{}...'.format(epoch))
        for video_name in training:
            model.train()
            visual_feature_dir = os.path.join(ground_dir,'parse_data',video_name)
            print("{} Training Start...".format(video_name))
        
            transcript = open(os.path.join(transcript_path,video_name+'_doc2vec.txt'),'r').readlines()
            transcript = [each.replace('/n','').replace('[','').replace(']','') for each in transcript]
            temp=[]
            for eachShot in transcript:
              temp.append([float(each) for each in eachShot.split(',')])
            transcript = torch.tensor(temp)
        
            features = representShot(visual_dir=visual_feature_dir,textual=transcript).to(device)
            groundtruth = load_gt(os.path.join(gt_path,video_name+'_middle_shot_attention.txt')).to(device)
            
            att_out = model(features)
            del features
            
            lossout = lossfun(att_out.view(-1,nshots),groundtruth)
            loss += lossout.item()
            lossout.backward()
            optimizer.zero_grad()
            optimizer.step()
        print("Loss :{}".format(loss/len(training)))
        
        if epoch % eval_rate == eval_rate-1:
            tmp = evaluate(model,video_list,mask=8)
            scheduler.step()
            if tmp>=f_score:
                f_score = tmp
                best_model = model
            print('Epoch {}, f_score: {}, best_fscore: {}'.format(epoch,tmp,f_score))
            
    if saved:
        model_save(best_model)
    else:
        return best_model
    

def evaluate(model,video_list,mask=30):
    """
    Evaluate model performance, evaluate model by list of video

    Parameters
    ----------
    model : model
        The model under test.
    video_list : list
        list of video name, model will get those video fscore.
    mask : int, optional
        Link in range -mask to mask are consider vaild link The default is 30

    Returns
    -------
    fscore : float
        F_score for this model.

    """
    print("Starting evaluating model...")
    ground_dir = '../'
    transcript_path = os.path.join(ground_dir,'transcript')
    nshots = 540
    fscore = 0
    record_point = [[] for i in range(len(video_list))]
    for i in range(len(video_list)):
        video_name = video_list[i]
        visual_feature_dir = os.path.join(ground_dir,'parse_data',video_name)
        # print("{} Evaluating Start...".format(video_name))
    
        transcript = open(os.path.join(transcript_path,video_name+'_doc2vec.txt'),'r').readlines()
        transcript = [each.replace('/n','').replace('[','').replace(']','') for each in transcript]
        temp=[]
        for eachShot in transcript:
          temp.append([float(each) for each in eachShot.split(',')])
        transcript = torch.tensor(temp)
    
        features = visual_feature(visual_dir=visual_feature_dir).to(device)
        
        att_out = model(features)
        _,pred = torch.topk(att_out.view(-1,nshots),5)
        boundary = pred_scenes(pred,mask=mask)
        score = fscore_eval(boundary,video_name)
        record_point[i].append(score)
        fscore +=score
    fscore = fscore/len(video_list)
    return fscore

def model_eval(model_path,mask=8):
    video_list = ['01_From_Pole_to_Pole','02_Mountains','03_Ice_Worlds','04_Great_Plains','05_Jungles','06_Seasonal_Forests','07_Fresh_Water',
                  '08_Ocean_Deep','09_Shallow_Seas','10_Caves','11_Deserts']
    model = MyTransformer(4396, 4, 6)
    model.load_state_dict(torch.load(model_path))
    f_score = evaluate(model, video_list,mask=mask)
    print("F_score: {}".format(f_score))
    
    
if __name__ == '__main__':
    # train_middle_shot(saved=True)
    # model_path = 'G:/model/20210319_model.pt'
    # model_eval(model_path)
    ground_dir = '../'
    video_list = ['01_From_Pole_to_Pole','02_Mountains','03_Ice_Worlds','04_Great_Plains','05_Jungles','06_Seasonal_Forests','07_Fresh_Water',
                  '08_Ocean_Deep','09_Shallow_Seas','10_Caves','11_Deserts']
    
    transcript_path = os.path.join(ground_dir,'transcript')
    gt_path = os.path.join(ground_dir,'annotations/scenes/annotator_1/')
    cuda = False
    check_file(video_list,ground_dir+'bbc_dataset_video')
    device = torch.device('cuda' if cuda else 'cpu')
    model = MyTransformer(4096,4,6)
    lossfun = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5)
    epoches =30
    eval_rate = 5
    nshots = 540
    f_score = 0
    for epoch in range(epoches):
        loss = 0
        # training, testing = train_test_split(video_list)
        print('Epoch :{}...'.format(epoch))
        for i in range(len(video_list)):
            video_name = video_list[i]
            model.train()
            visual_feature_dir = os.path.join(ground_dir,'parse_data',video_name)
            print("{} Training Start...".format(video_name))
        
            # transcript = open(os.path.join(transcript_path,video_name+'_doc2vec.txt'),'r').readlines()
            # transcript = [each.replace('/n','').replace('[','').replace(']','') for each in transcript]
            # temp=[]
            # for eachShot in transcript:
            #   temp.append([float(each) for each in eachShot.split(',')])
            # transcript = torch.tensor(temp)
        
            features = visual_feature(visual_dir=visual_feature_dir).to(device)
            groundtruth = load_keyShot(video_name).to(device)
            
            att_out = model(features)
            del features
            
            lossout = lossfun(att_out.view(-1,nshots),groundtruth)
            loss += lossout.item()
            lossout.backward()
            optimizer.zero_grad()
            optimizer.step()
            print("Epoch {}, loss: {}".format(epoch,loss/(i+1)))
        
        if epoch % eval_rate == eval_rate-1:
            tmp = evaluate(model,video_list,mask=8)
            scheduler.step()
            if tmp>=f_score:
                f_score = tmp
                best_model = model
            print('Epoch {}, f_score: {}, best_fscore: {}'.format(epoch,tmp,f_score))
    model_save(best_model,save_path='G:/model')
             