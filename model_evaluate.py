# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:06:15 2021

@author: NTUT
"""

import os
import torch
from torch import nn
import numpy as np


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
    
def load_model(model,model_name=None):
    """
    Parameters
    ----------
    model : Transformer
        model must be create before loading
    model_name : str, optional
        model_name must descript absolute path for model. The default is None.

    Returns
    -------
    None.

    """
    if model_name is None:
        model_name = os.listdir('G:/model')[0]
    
    if not model_name.endswith(('.pt','.pth')):
        raise TypeError("Excepted type .pt or .pth, got {} instance.".format(model_name))
    model.load_state_dict(torch.load(model_name))

def load_transcrpit(video_name):
    """
    Parameters
    ----------
    video_name : str
    Returns
    -------
    transcript : torch.tensor
        return transcript as doc2vec in type of torch.tensor

    """
    transcript_dir = os.path.join('../','transcript')
    transcript = open(os.path.join(transcript_dir,video_name+'_doc2vec.txt'),'r').readlines()
    transcript = [each.replace('\n','').replace('[','').replace(']','') for each in transcript]
    temp = []
    for eachShot in transcript:
        temp.append([float(each) for each in eachShot.split(',')])
    transcript = torch.tensor(temp)
    return transcript

def representShot(visual_dir,textual):
    listShots = os.listdir(visual_dir)
    nShot = len(listShots)
    result = torch.empty((nShot,4396))
    
    assert textual.shape[0] == nShot
    for i in range(nShot):
        visual_feature_path = os.path.join(visual_dir,listShots[i])
        if not os.path.isfile(visual_feature_path):
            raise RuntimeError("{} not exist, check file location".format(visual_feature_path))
        elif not visual_feature_path.endswith(('.pt','pth')):
            raise TypeError("Excepted type .pt or .pth, got {} instance.".format(visual_feature_path))
        else:
            # print("Loading {}".format(visual_feature_path.replace('.pt','')))
            visual = torch.load(visual_feature_path,map_location=torch.device('cpu'))
            text = textual[i]
            shot_feature = torch.cat((visual.view(1,-1),text.view(1,-1)),1)
            result[i,:] = shot_feature
    print("Load finish, result shape: {}".format(result.shape))
    return result

def save_attention_out(out,name,path='G:/attention_result'):
    """
    Parameters
    ----------
    out : torch.tensor
        Attention Result to Save
    name : str
        Save name, end with .pt or .pth are not required.\nSave name are as type "Attention_out_name.pt"
    path : TYPE, optional
        Save path. The default is 'G:/attention_result'.

    Returns
    -------
    None.
    
    """
    name = 'Attention_out_'+name
    if name.endswith(('.pt','.pth')):
        save_path = os.path.join(path,name)
    else:
        save_path = os.path.join(path,name+'.pt')
    torch.save(out,save_path)
    print("Saving Result {}".format(save_path))
    
if __name__ == '__main__':
    model = MyTransformer(4396, 4, 6)
    load_model(model,model_name='G:/model/20210217_epoch4_model.pt')
    ground_dir = '../'
    video_list = ['01_From_Pole_to_Pole','02_Mountains','05_Jungles','06_Seasonal_Forests',
            '08_Ocean_Deep','09_Shallow_Seas','10_Caves','11_Deserts']
    
    for video_name in video_list:
        visual_dir = os.path.join(ground_dir,'parse_data',video_name)
        textual = load_transcrpit(video_name)
        print("Processing {}".format(video_name))
        
        feature = representShot(visual_dir, textual)
        final_out, att_out = model(feature)
                
        save_attention_out(att_out, video_name)
        
    
    
