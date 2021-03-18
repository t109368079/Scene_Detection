# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 10:13:31 2021

@author: NTUT
"""

import os
import torch 
from torch import nn
from torch import optim
from torch.nn import functional as F
import random
import numpy as np
from coverage_overflow import coverOverflow


class boundaryDetector(nn.Module):
    def __init__(self,dim):
        super(boundaryDetector,self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(dim,2048)
        self.layer2 = nn.Linear(2048,2048)
        self.layer3 = nn.Linear(2048,1024)
        self.layer4 = nn.Linear(1024,2)
    
    def forward(self,attention):
        x = F.relu(self.layer1(attention))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return x

def load_gt(video_name):
    """

    Parameters
    ----------
    video_name : str
        groundtruth file are a series of 0 and 1. 1 is represented for boundary

    Returns
    -------
    gt : torch.tensor

    """
    gt_dir = '../annotations/scenes/annotator_0/'
    gt_name = os.path.join(gt_dir,video_name+'_BGT.txt')
    tmp = open(gt_name,'r').readlines()
    tmp = [each.replace('\n','').replace('[','').replace(']','') for each in tmp]
    gt = []
    for each in tmp[0].split(','):
        gt.append(int(each))
    gt = torch.tensor(gt)
    return gt
    

def train_test_split(dataset,test_size=0.3):
    nvideo = len(dataset)
    ntest = int(test_size*nvideo)
    index = [i for i in range(nvideo)]
    random.shuffle(index)
    training = [dataset[i] for i in index[ntest:]]
    testing = [dataset[i] for i in index[:ntest]]
    return training, testing

def accfun(pred,label):
    acc = 0
    for p, l in zip(pred,label):
        if p == l:
            acc += 1
    return acc/pred.shape[0]


def convert(gt):
    """
    It will convert a sequence of 0,1 to shot index
    Parameters
    ----------
    gt : np.array
        gt can be either groundtruth ot prediction.

    Returns
    -------
    new_gt: np.array
        Boundary shot index will be presented in new_gt
    """
    
    boundary = []
    for i in range(len(gt)):
        if gt[i] == 1:
            boundary.append(i)
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
    
    
if __name__ == '__main__':
    video_list = ['01_From_Pole_to_Pole','02_Mountains','05_Jungles','06_Seasonal_Forests',
            '08_Ocean_Deep','09_Shallow_Seas','10_Caves','11_Deserts']
    
    training, testing = train_test_split(video_list)    
    boundary_model = boundaryDetector(4396)
    class_weight = torch.tensor([1,10],dtype=torch.float32)
    class_weight = class_weight/class_weight.sum()
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = optim.SGD(boundary_model.parameters(),lr=0.001,momentum=0.9)
    # Training...
    for epoch in range(10):
        training_loss=0
        for video_name in training: 
            pt_name = 'Attention_out_'+video_name+'.pt'
            video_gt = load_gt(video_name)
            attention = torch.load(os.path.join('G:/','attention_result',pt_name))
        
            out = boundary_model(attention)
            # video_gt = video_gt.float()
            loss = criterion(out,video_gt)
            training_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("Epoch {}\t Training_loss {}".format(epoch+1,training_loss))
        # Vaild...
        for video_name in testing:
            video_gt = load_gt(video_name)
            pt_name = 'Attention_out_'+video_name+'.pt'
            attention = torch.load(os.path.join('G:/','attention_result',pt_name))
            out = boundary_model(attention)
            _, boundary = torch.max(out,1)
            acc = accfun(boundary,video_gt)
            print("Epoch {}\tVideo {} Accuracy: {}".format(epoch,video_name,acc))
            
    #Predict
    for video_name in video_list:
        video_gt = load_gt(video_name)
        video_gt = video_gt.detach().numpy()
        pt_name = 'Attention_out_'+video_name+'.pt'
        attention = torch.load(os.path.join('G:/','attention_result',pt_name))
        out = boundary_model(attention)
        _, boundary = torch.max(out,1)
        boundary = boundary.detach().numpy()
        coverage, overflow = coverOverflow(video_gt,boundary)
        fscore = scoring(coverage,overflow,printed=True,string=video_name)
        boundary = convert(boundary)
        file = open(os.path.join('../result/scene_boundary_result/',video_name+'_boundary.txt'),'w')
        file.write(str(boundary))
        # print("Finish Prediction {}".format(video_name))
        file.close()
            
        
