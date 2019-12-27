# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:25:37 2019

@author: daewo
"""

import os
import math

import numpy as np

#from LeNet_Class import LeNet
#from LeNet_Class import DataGen
#from LeNet_Class import ResetKeras

from UNet_Class import UNet
from UNet_Class import DataGen
from UNet_Class import ResetKeras

#from VGG_Class import VGG
#from VGG_Class import DataGen
#from VGG_Class import ResetKeras

#%% Read Images

# Walk into directories in filesystem
# Ripped from os module and slightly modified
# for alphabetical sorting
def sortedWalk(top, topdown=True, onerror=None):
    from os.path import join, isdir
 
    names = os.listdir(top)
    names.sort()
    dirs, nondirs = [], []
 
    for name in names:
        if isdir(os.path.join(top, name)):
            dirs.append(name)
        else:
            nondirs.append(name)
 
    if topdown:
        yield top, dirs, nondirs
    for name in dirs:
        path = join(top, name)
        if not os.path.islink(path):
            for x in sortedWalk(path, topdown, onerror):
                yield x
    if not topdown:
        yield top, dirs, nondirs


#%% Define paramenters

#LeNet
#n_classes = 2
#size_patch = 28

#UNet
image_size = 128

#FCN_VGG
#n_classes = 2
#image_size = 128

#%% Get Data 

train_path = "../Datos/bus/"

#Get the images file name
train_ids = next(sortedWalk(train_path))[2]

#Set the data generator parameters
gen = DataGen(train_ids, train_path) #LeNet
gen = DataGen(train_ids, train_path, image_size = image_size) #UNet, VGG

#Read all images and its masks
[tumor_images, mask_images] = gen.__load_all__()
    
#%% Cross validation K-fold

n_imgs = len(tumor_images)
k = 30
n_test_imgs = math.floor(n_imgs/k)

for i in range(k):
#for i in range(2):
    
    fold_start = i*n_test_imgs
    fold_end = fold_start + n_test_imgs - 1
    
    test_images = tumor_images[fold_start:fold_end]
    test_masks = mask_images[fold_start:fold_end]
    
    train_images = np.delete(tumor_images,slice(fold_start,fold_end),axis=0)
    train_masks = np.delete(mask_images,slice(fold_start,fold_end),axis=0)
    
    #%% train model
    
#    arch = LeNet()
    arch = UNet()
#    arch = VGG()
    
    #Create model
#    model = arch.modelArch(size_patch=size_patch, n_classes=n_classes) #LeNet
    model = arch.modelArch(image_size=image_size) #UNet
#    model = arch.modelArch(n_classes=n_classes, image_size=image_size) #VGG
    
    #compile model
    model = arch.compileModel(model=model)
    
#    train model
#    model = arch.trainModel(train_images,train_masks,model,size_patch,epochs=100,fold=i) #LeNet
    model = arch.trainModel(train_images,train_masks,model,image_size=image_size,epochs=100,fold=i) #UNet, VGG

    
    ResetKeras(model)