# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 10:34:18 2020
Lala~這是用來測試特徵的py檔
用一句話總結，feature都很像!!

@author: garyy
"""

import csv
import torch 
import numpy as np
import pandas as pd
import torch
from torch import nn 
from self_attention_new import SelfAttention
from matplotlib import pyplot as plt

def representShots(shots, textual):
    result = []
    for indexOfShots in range(len(shots)):
        eachShot = torch.tensor(shots[indexOfShots])
        temp = torch.mean(eachShot, dim=0)
        featureWithText = torch.cat((temp.view(1, -1), textual[indexOfShots].view(1, -1)), 1)
        result.append(featureWithText)
    result = torch.cat(result)
    return result

def GetShots(keyframe_path, inputPath):
    input_file = open(keyframe_path, "r")
    rows = csv.reader(input_file)
    rows = list(rows)
    input_file.close()
    shots = sorted(set(map(lambda x: x[0], rows)))
    shotsList = [[eachRow[1] for eachRow in rows if eachRow[0] == shotIndex] for shotIndex in shots]
    shotsInVideo = []
    features = torch.load(inputPath,map_location=torch.device('cpu'))
    for eachShot in shotsList:
        eachShot.sort()
        framesInEachShot = []
        for indexOfFrame in range(5):            
            framesInEachShot.append(features[:,int(eachShot[indexOfFrame])].tolist())
        shotsInVideo.append(framesInEachShot)
    return shotsInVideo

class myModel(nn.Module):
    def __init__(self, frame_dim, hidden_dim, output_dim):
        super(myModel, self).__init__()
        self.hidden_dim = hidden_dim
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(frame_dim, hidden_dim // 2, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, output_dim)
        self.attention = SelfAttention(1, 2348, 2348, 2348)
        self.layer = nn.Linear(2348, 2348)
    
    def forward(self, shotsFeature, textual, token, index=0):

        # length here means the number of shots in the video
        length = shotsFeature.shape[0]
        batch_shotsFeature = shotsFeature.view(1, length, -1)
        Y, _ = self.attention(batch_shotsFeature, batch_shotsFeature, batch_shotsFeature)
        del batch_shotsFeature

        # record 
        # if token == 1:
        #     torch.save(shotsFeature, "../result/20200708_new_average_res_key_cos_y_video" + str(index) + ".pt")
        #     torch.save(Y, "../result/20200708_new_attention_res_key_cos_average_y_video" + str(index) + ".pt")

        finalRsult = []
        Y = Y.view(Y.shape[1], -1)

        for eachAttentionResult in Y:
            tempResult = []
            for eachShot in Y:
                tempResult.append(torch.dot(eachAttentionResult, eachShot))
            finalRsult.append(tempResult)
        del Y
        del shotsFeature
        predictResult = torch.tensor(finalRsult, requires_grad = True)
        return predictResult
video_name = '01_From_Pole_to_Pole'
video_id = 'bbc_01'
feature_path = '../parse_data/'+video_id+'_feature_resnet_101.pt'
transcript_path = '../transcript/'+video_name+'_doc2vec.txt'
keyframe_path = '../annotations/shots/keyframes/'+video_name[:2]+'_output.csv'
shot_path = '../annotations/shots/'+video_name+'.txt'

shots = GetShots(keyframe_path, feature_path)
transcriptList = open(transcript_path,'r').readlines()
transcriptList = [each.replace('\n','').replace('[','').replace(']','') for each in transcriptList]
temp = []
for eachShot in transcriptList:
    temp.append([float(each) for each in eachShot.split(", ")])
transcript = torch.tensor(temp)
representativeFeatures = representShots(shots, transcript)
shot_feature = representativeFeatures.detach().numpy()

# corr = np.zeros((445,445))
# for i in range(shot_feature.shape[0]):
#     source_feature = shot_feature[i,:]
#     for j in range(i,shot_feature.shape[0]):
#         target_feature = shot_feature[j,:]
#         corr[i,j] = np.dot(source_feature,target_feature)/(np.linalg.norm(source_feature)*np.linalg.norm(target_feature))

groundtruth = open('../annotations/scenes/annotator_1/'+video_name+'_NGT.txt','r').readlines()
groundtruth = [each.replace('\n','').replace('[','').replace(']','') for each in groundtruth]     
temp = []
for each in groundtruth:
    temp.append([int(num) for num in each.split(',')])
groundtruth = temp   
label = torch.tensor(temp)
model = myModel(2048, 2048, 2048)
transformer = nn.Transformer(d_model=2348,nhead=4)
optimizer = torch.optim.Adam(transformer.parameters(),lr=0.001)
lossfun = torch.nn.BCEWithLogitsLoss()
record_point = []

## Training on our model

for i in range(10):
    print('Training on epoch: '+str(i))
    out = model(representativeFeatures,transcript,0)
    attention_result = out.detach().numpy()
    temp = attention_result[8,:]/max(attention_result[8,:])
    record_point.append(temp)
    label = label.type_as(out)
    loss = lossfun(out,label)
    loss.backward()
    optimizer.zero_grad()
    optimizer.step()
    print("Training loss: {}".format(loss.item()))
 

plt.figure()
plt.title("bbc_01 selected scene")
plt.plot(groundtruth[8][0:40],label='Ground Truth')
for i in range(len(record_point)):
    plt.plot(record_point[i][0:40],label='epoch{}'.format(i))
plt.legend(loc='lower right')
plt.xlabel("Shot index")
plt.ylabel("Correlation")
plt.show()            
 
## Training on transformer with pytorch
## 12/28 problem is shape of src(S,N,E) and tgt(T,N,E) E shoudl equal to d_model
## How to view src and tgt?
# for i in range(1):
#     print("Training on epoch: "+str(i))
#     seq_len,feature_len = representativeFeatures.shape
#     src = representativeFeatures.view(seq_len,1,feature_len)
#     tgt = label.view(seq_len,1,seq_len)
#     print(src.shape)
#     print(tgt.shape)
#     out = transformer(src,src)
#     attention_result = out.detach().numpy()
#     temp = attention_result[8,:]/max(attention_result[8,:])
#     record_point.append(temp)
#     label = label.type_as(out)
#     loss = lossfun(label,out)
#     loss.backward()
#     optimizer.step()
#     print("Training loss: {}".format(loss.item()))

plt.figure()
plt.title("bbc_01 selected scene")
plt.plot(groundtruth[8][0:40],label='Ground Truth')
for i in range(len(record_point)):
    plt.plot(record_point[i][0:40],label='epoch{}'.format(i))
plt.legend(loc='lower right')
plt.xlabel("Shot index")
plt.ylabel("Correlation")
plt.show()  