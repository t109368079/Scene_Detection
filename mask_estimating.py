# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:08:02 2021

@author: NTUT
"""
import time
import random
import os
import csv
import random
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from coverage_overflow import load_bgt, coverOverflow, fscore_eval

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F


class MyTransformer(nn.Module):
    def __init__(self,d_model,nhead,num_layer):
        super(MyTransformer,self).__init__()
        #Model Parameter
        self.d_model = d_model
        self.nhead = nhead
        self.num_layer = num_layer
        #Model Architecture
        encoderLayer = nn.TransformerEncoderLayer(self.d_model, self.nhead)
        self.encoder = nn.TransformerEncoder(encoderLayer,self.num_layer)
        self.layer1 = nn.Linear(d_model,2048)
        self.layer2 = nn.Linear(2048,1024)
        self.layer3 = nn.Linear(1024,540)
    
    def forward(self,shot_feature):
        num_shot = shot_feature.shape[0]
        src = shot_feature.view(num_shot,1,self.d_model)
        attention_out = self.encoder(src)
        
        x = F.relu(self.layer1(attention_out))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        
        return x
    
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
      save_path = '../model'
    
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

def pred_scenes(pred,mask=30):
    """

    Parameters
    ----------
    pred : torch.tensor
        pred are top 5 shot current shot attention to.
    mask : int, optional
        In pred, only care about the shot in range current index-mask to current index+mask The default is 30.

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
        start = min(link)
        end = max(link)
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

def mask_estimate(model,video_list):
    print("Starting evaluating model...")
    ground_dir = '../'
    transcript_path = os.path.join(ground_dir,'transcript')
    nshots = 540
    fscore = 0
    new_fscore = 0
    score_record_point = [[] for i in range(len(video_list))]
    cover_record_point = [[] for i in range(len(video_list))]
    overf_record_point = [[] for i in range(len(video_list))]
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
    
        features = representShot(visual_dir=visual_feature_dir,textual=transcript)
        
        att_out = model(features)
        _,pred = torch.topk(att_out.view(-1,nshots),5)
        for mask in range(5,31):
            boundary = pred_scenes(pred,mask=mask)
            score = fscore_eval(boundary,video_name,coverover=False)
            score_record_point[i].append(score)
        # cover_record_point[i].append(np.mean(cover))
        # overf_record_point[i].append(np.mean(over))
        fscore +=score
    fscore = fscore/len(video_list)
    if fscore >= new_fscore:
        new_fscore = fscore
        best_mask = mask
    return new_fscore, best_mask, score_record_point
    

if __name__ == '__main__':
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
    epoches = 51
    eval_rate = 5
    nshots = 540
    new_fscore = 0
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
        
        if epoch % eval_rate == 0:
            score_record_point = [[] for i in range(len(video_list))]
            cover_record_point = [[] for i in range(len(video_list))]
            overf_record_point = [[] for i in range(len(video_list))]
            for i in range(len(video_list)):
                model.eval()
                video_name = video_list[i]
                visual_feature_dir = os.path.join(ground_dir,'parse_data',video_name)
                # print("{} Evaluating Start...".format(video_name))
            
                transcript = open(os.path.join(transcript_path,video_name+'_doc2vec.txt'),'r').readlines()
                transcript = [each.replace('/n','').replace('[','').replace(']','') for each in transcript]
                temp=[]
                for eachShot in transcript:
                  temp.append([float(each) for each in eachShot.split(',')])
                transcript = torch.tensor(temp)
            
                features = representShot(visual_dir=visual_feature_dir,textual=transcript)
                
                att_out = model(features)
                _,pred = torch.topk(att_out.view(-1,nshots),5)
                for mask in range(5,31):
                    boundary = pred_scenes(pred,mask=mask)
                    cover, over, score = fscore_eval(boundary,video_name,coverover=True)
                    score_record_point[i].append(score)
                    cover_record_point[i].append(np.mean(cover))
                    overf_record_point[i].append(np.mean(over))
            score_record_point = np.array(score_record_point)
            f_score_list = np.mean(score_record_point,axis=0)
            fscore = max(f_score_list)
            mask = np.argmax(f_score_list)
            if fscore >= new_fscore:
                new_fscore = fscore
                best_mask = mask
                best_model = model
            print('Epoch {}, f_score: {}, best_fscore: {}, mask size: {}'.format(epoch,fscore,new_fscore,best_mask))
        
        for i in range(len(video_list)):
            plt.figure()
            plt.title('{} Coverage'.format(video_list[i]))
            plt.plot(cover_record_point[i],label='coverage')
            plt.plot(overf_record_point[i],label='overflow')
            plt.plot(f_score_list,label='fscore')
            plt.legend()
            plt.xlabel('mask size')
            plt.ylabel('Score')
            plt.savefig('G:/plot/{}.png'.format(video_list[i]))
            plt.show()