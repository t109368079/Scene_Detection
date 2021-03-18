# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 10:20:56 2020

@author: garyy
"""

import cv2
import csv
import torch
import numpy as np
from torch import nn
from torch import optim
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

def GetShots(indexOfVideo, inputPath):
    input_folder = "../annotations/shots/keyframes/"
    file_name = numberToString(indexOfVideo + 1) + "_output.csv"
    input_file = open(input_folder + file_name, "r")
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

def numberToString(index):
    if index < 10:
        return "0" + str(index)
    return str(index)

class MyTransformer(nn.Module):
    def __init__(self,num_shot,d_model,nhead,num_layers):
        super(MyTransformer, self).__init__()
        # Model parameter 
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_shot = num_shot
        # Model architecture
        self.transform = nn.Transformer(d_model=d_model,nhead=nhead,num_encoder_layers=num_layers,num_decoder_layers=num_layers)
    
    def forward(self,shot_feature):
        # shot_feature must be 2-D tensor(#shot,#feature)
        src = shot_feature.view(self.num_shot,1,self.d_model)
        tgt = src
        attention_out = self.transform(src,tgt)
        attention_out = attention_out.view(self.num_shot,self.d_model)
        
        # print('Shape of attention result: {}'.format(attention_out.shape))
        result =np.zeros((self.num_shot,self.num_shot))
        i = 0
        for src in attention_out:
            temp = []
            src_norm = torch.norm(src)    # dim = 1 means 橫的方向
            for tgt in attention_out:
                tgt_norm = torch.norm(tgt)
                corr = torch.dot(src,tgt)/(src_norm*tgt_norm)
                corr = corr.detach().numpy()
                temp.append(corr)
            temp = np.array(temp)
            result[i,:] = temp
            i = i+1
            
        final_result = torch.tensor(result,requires_grad=True)
        del result
        return final_result ,attention_out
    

if __name__ == '__main__':
    feature_path_list = ['../parse_data/bbc_01_feature_resnet_101.pt', '../parse_data/bbc_02_feature_resnet_101.pt', '../parse_data/bbc_03_feature_resnet_101.pt', '../parse_data/bbc_04_feature_resnet_101.pt', '../parse_data/bbc_05_feature_resnet_101.pt', '../parse_data/bbc_06_feature_resnet_101.pt', '../parse_data/bbc_07_feature_resnet_101.pt', '../parse_data/bbc_08_feature_resnet_101.pt', '../parse_data/bbc_09_feature_resnet_101.pt', '../parse_data/bbc_10_feature_resnet_101.pt', '../parse_data/bbc_11_feature_resnet_101.pt']
    inputShots = "../annotations/shots/"
    inputPathOfTranscript = "../transcript/"
    groundtruthFolder = "../annotations/scenes/annotator_0/"
    videoName_list = ["01_From_Pole_to_Pole.txt", "02_Mountains.txt", "03_Ice Worlds.txt", "04_Great Plains.txt", "05_Jungles.txt", "06_Seasonal_Forests.txt", "07_Fresh_Water.txt", "08_Ocean_Deep.txt", "09_Shallow_Seas.txt", "10_Caves.txt", "11_Deserts.txt"]
    # First I just train on one video
    feature_path = feature_path_list[0]
    videoName = videoName_list[0]
    groundtruth = open(groundtruthFolder+videoName.replace('.txt','_NGT.txt'),'r').readlines()
    groundtruth = [each.replace('\n','').replace('[','').replace(']','') for each in groundtruth]
    temp=[]
    for shot in groundtruth:
        temp.append([int(each) for each in shot.split(',')])
    groundtruth = temp
    label = torch.tensor(temp)

    
    shots = GetShots(0, feature_path)
    
    transcriptList = open(inputPathOfTranscript + videoName.replace(".txt", "_doc2vec.txt"), "r").readlines()
    transcriptList = [each.replace("\n", "").replace("[", "").replace("]", "") for each in transcriptList]
    temp = []
    for eachShot in transcriptList:
        temp.append([float(each) for each in eachShot.split(", ")])
    transcript = torch.tensor(temp)
    features = representShots(shots, transcript)

    model = MyTransformer(features.shape[0],features.shape[1],nhead=4,num_layers=4)
    lossfun = nn.MSELoss()
    opti = optim.Adam(model.parameters(),lr=0.001)
    epoch = 50
    record_point=[]
    while(epoch>0):
        final_out, attention_out = model(features)
        attention_result = final_out.detach().numpy()
        temp = attention_result[8,:]/max(attention_result[8,:])
        record_point.append(temp)
        label = label.type_as(final_out)
        loss = lossfun(label,final_out)
        loss.backward()
        opti.step()
        print("Training loss: {}".format(loss.item()))
        epoch = epoch-1

plt.figure()
plt.title("bbc_01 selected scene-attention")
plt.plot(groundtruth[8][0:20],label='Ground Truth')
for i in range(len(record_point)):
    if(i == 0):
        plt.plot(record_point[i][0:20],label='epoch{}'.format(i+1))
    if(i%10 == 9):
        plt.plot(record_point[i][0:20],label='epoch{}'.format(i+1))
plt.legend(loc='lower right')
plt.xlabel("Shot index")
plt.ylabel("Correlation")
plt.show()    
    
    
    