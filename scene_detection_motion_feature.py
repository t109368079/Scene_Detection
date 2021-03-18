# -*- coding: utf-8 -*-
"""Scene_Detection_motion_feature.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cOck9u0mY8tSCVv4W-D-9khHd8Zh3VCR
"""

import time
import random
import os
import csv
import random
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
from datetime import datetime

import torch
from torch import nn
from torch import optim

class MyTransformer(nn.Module):
    def __init__(self,d_model,nhead,num_layer):
        super(MyTransformer,self).__init__()
        #Model Parameter
        self.d_model = d_model
        self.nhead = nhead
        self.num_layer = num_layer
        #Model Architecture
        self.transformer = nn.Transformer(d_model=d_model,nhead=nhead,num_encoder_layers=num_layer,num_decoder_layers=num_layer)
    
    def forward(self,shot_feature):
        num_shot = shot_feature.shape[0]
        src = shot_feature.view(num_shot,1,self.d_model)
        tgt = src
        attention_out = self.transformer(src,tgt)
        attention_out = attention_out.view(num_shot,self.d_model)
        
        result = np.empty((num_shot,num_shot))
        i = 0
        for src in attention_out:
            
          temp = []
          src_norm = torch.norm(src)
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
        return final_result, attention_out

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
    print("Load finish, result shape: {}".format(result.shape))
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

def accfun(result,label):
    temp = result*label
    temp = temp.view(1,-1)
    acc = torch.sum(temp).detach().numpy()
    
    return acc

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

def clean_console():
    print("\014")

if __name__ == '__main__':
    ground_dir = '../'
    video_list = ['01_From_Pole_to_Pole','02_Mountains','05_Jungles','06_Seasonal_Forests',
            '08_Ocean_Deep','09_Shallow_Seas','10_Caves','11_Deserts']
    
    transcript_path = os.path.join(ground_dir,'transcript')
    
    check_file(video_list,ground_dir+'bbc_dataset_video')
    record_point = []
    model = MyTransformer(4396,nhead=4,num_layer=6)
    lossfun = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    for epoch in range(5):
      acc = 0
      loss = 0
      training, testing = train_test_split(video_list)
      # Training...
      for video_name in training:
        visual_feature_dir = os.path.join(ground_dir,'parse_data',video_name)
        print("{} Training Start...".format(video_name))
    
        transcript = open(os.path.join(transcript_path,video_name+'_doc2vec.txt'),'r').readlines()
        transcript = [each.replace('/n','').replace('[','').replace(']','') for each in transcript]
        temp=[]
        for eachShot in transcript:
          temp.append([float(each) for each in eachShot.split(',')])
        transcript = torch.tensor(temp)
    
        features = representShot(visual_dir=visual_feature_dir,textual=transcript)
        # print("Feature shape: {}".format(features.shape))
    
        gt_path = ground_dir+'/annotations/scenes/annotator_0'
        groundtruth = open(os.path.join(gt_path,video_name+'_NGT.txt'),'r').readlines()
        groundtruth = [each.replace('/n','').replace('[','').replace(']','') for each in groundtruth]
        temp = []
        for gt_shot in groundtruth:
          temp.append([float(each) for each in gt_shot.split(',')])
        groundtruth = temp
        label = torch.tensor(groundtruth)
        del temp
    
    
        # sched = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.5)
        final_out,att_out = model(features)
        del features
        attention_result = final_out.detach().numpy()
        temp = attention_result[8,:]/max(attention_result[8,:])
        record_point.append(temp)
        label = label.type_as(final_out)
        lossout = lossfun(final_out,label)
        loss += lossout.item()
        acc += accfun(final_out,label)
        lossout.backward()
        optimizer.step()
      print("Epoch {}...\t Training Acc: {:.3f}\t Training loss: {:.3f}".format(epoch+1,acc,loss))
      # Testing...
      test_acc = 0
      for video_name in testing:
        visual_feature_dir = os.path.join(ground_dir,'parse_data',video_name)
        print("{} Testing Strat...".format(video_name))
    
        transcript = open(os.path.join(transcript_path,video_name+'_doc2vec.txt'),'r').readlines()
        transcript = [each.replace('/n','').replace('[','').replace(']','') for each in transcript]
        temp=[]
        for eachShot in transcript:
          temp.append([float(each) for each in eachShot.split(',')])
        transcript = torch.tensor(temp)
    
        features = representShot(visual_dir=visual_feature_dir,textual=transcript)
        clear_output()
        print("Feature shape: {}".format(features.shape))
    
        gt_path = ground_dir+'/annotations/scenes/annotator_0'
        groundtruth = open(os.path.join(gt_path,video_name+'_NGT.txt'),'r').readlines()
        groundtruth = [each.replace('/n','').replace('[','').replace(']','') for each in groundtruth]
        temp = []
        for gt_shot in groundtruth:
          temp.append([float(each) for each in gt_shot.split(',')])
        groundtruth = temp
        label = torch.tensor(groundtruth)
        del temp
    
        final_out, _ = model(features)
        del features
        label = label.type_as(final_out)
        test_acc += accfun(final_out,label)
      print("Epoch {}...\t Testing Accuracy: {:.3f}".format(epoch+1,test_acc))
      model_save(model,epoch=epoch,save_path='E:/model')

# for i in range(len(record_point)):
#   plt.figure()
#   plt.title("bbc_01 select scene-attention")
#   plt.plot(groundtruth[8][0:50],label='Ground truth')
#   plt.plot(record_point[i][0:50],label='Epoch{}'.format(i))
#   plt.legend(loc='lower right')
#   plt.xlabel('Shot index')
#   plt.ylabel('Correlation')
#   plt.show()

# temp

