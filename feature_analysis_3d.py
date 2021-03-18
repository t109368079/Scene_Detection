# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 15:22:46 2021

Feature analysis and feature enhence
@author: NTUT
"""

import os
import numpy as np
import pandas as pd

import torch
from torch import nn
from matplotlib import pyplot as plt


video_name = '01_From_Pole_to_Pole'
data_dir = os.path.join('../parse_data/',video_name)

f = os.listdir(data_dir)
nShot = len(f)
feature_map = np.empty((nShot,4096))
for i in range(nShot):
    load_name = os.path.join(data_dir,'shot_{}.pt'.format(str(i)))
    tmp = torch.load(load_name)
    feature = tmp.detach().numpy()
    feature_map[i] = feature

corr = np.corrcoef(feature_map)


