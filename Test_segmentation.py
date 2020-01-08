# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 11:21:10 2019

@author: daewo
"""
import os
import math

import numpy as np

#from LeNet_Class import LeNet
#from LeNet_Class import DataGen
#from LeNet_Class import ResetKeras

#from UNet_Class import UNet
#from UNet_Class import DataGen
#from UNet_Class import ResetKeras

from VGG_Class import VGG
from VGG_Class import DataGen
from VGG_Class import ResetKeras

from PostProcessing_Class import PostProcessing

#%% Read Images

# Walk into directories in filesystem
# Ripped from os module and slightly modified
# for alphabetical sorting
#
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
#image_size = 128

#FCN_VGG
n_classes = 2
image_size = 256

#%% Get Data 
#train_path = "../Datos/bus/"
train_path = "../Datos/Img/"

#Get the images file name
train_ids = next(sortedWalk(train_path))[2]

#Set the data generator parameters
#gen = DataGen(train_ids, train_path) #LeNet
gen = DataGen(train_ids, train_path, image_size = image_size) #UNet, VGG

#Read all images and its masks
[tumor_images, mask_images] = gen.__load_all__()
#%% Cross validation K-fold

n_imgs = len(tumor_images)
k = 10
n_test_imgs = math.floor(n_imgs/k)

accuracy = []
sensitivity = []
specificity = []
roc_au = []

for i in range(k):
#for i in range(2):
    
    fold_start = i*n_test_imgs
    fold_end = fold_start + n_test_imgs - 1
    
    test_images = tumor_images[fold_start:fold_end]
    test_masks = mask_images[fold_start:fold_end]
    
    train_images = np.delete(tumor_images,slice(fold_start,fold_end),axis=0)
    train_masks = np.delete(mask_images,slice(fold_start,fold_end),axis=0)
    
    #%% Test model
    

    
#    arch = LeNet()
#    arch = UNet()
    arch = VGG()
    
    #Create model
#    model = arch.modelArch(size_patch=size_patch, n_classes=n_classes) #LeNet
#    model = arch.modelArch(image_size=image_size) #UNet
    model = arch.modelArch(n_classes=n_classes, image_size=image_size) #VGG
    arch.compileModel(model)
    results_path = arch.results_path
    
    #Load saved weights for fold
#    train_file = "LeNet_Weights_fold_" + str(i) + ".h5"
#    train_file = "UNet_Weights_fold_" + str(i) + ".h5"
    train_file = "VGG_Weights_fold_" + str(i) + ".h5"
    train_dir = os.path.join(results_path,"train_weights",train_file)
    model.load_weights(train_dir)
    
    #Predict the probability of each pixel
#    [prob_images, class_images] = arch.predictClassModel(test_images,model=model,size_patch=size_patch) #LeNet
    [prob_images, class_images] = arch.predictClassModel(test_images,model=model,image_size=image_size) #UNet, VGG
    
    #Visualize and evaluate predictions
    post_process = PostProcessing()
    
#    post_process.visualize(test_images,class_images,prob_images,save_path=results_path,fold=i,n_disp=1) #LeNet
    post_process.visualize(test_images[:,:,:,0],class_images,prob_images,save_path=results_path,fold=i,n_disp=5) #UNeT VGG
    
#    [acc_fold,sen_fold,spec_fold,auc_fold] = post_process.evaluateSegmentation(test_masks,class_images,save_path=results_path,fold=i) #LeNet
    [acc_fold,sen_fold,spec_fold,auc_fold] = post_process.evaluateSegmentation(test_masks[:,:,:,0],class_images,save_path=results_path,fold=i) #UNeT VGG
#
    accuracy.append(acc_fold[-1])
    sensitivity.append(sen_fold[-1])
    specificity.append(spec_fold[-1])
    roc_au.append(auc_fold[-1])
    
    ResetKeras(model)
    
    
#Write segmentation results for all folds
results_filename = "results_total.txt"
results_filename = os.path.join(results_path,"ROC",results_filename)
results_file= open(results_filename,"w")
results_file.write("Accuracy: " + str(sum(accuracy)/len(accuracy)) + "\n")
results_file.write("Sensitivity: " + str(sum(sensitivity)/len(sensitivity)) + "\n")
results_file.write("Specificity: " + str(sum(specificity)/len(specificity)) + "\n")
results_file.write("AU-ROC: " + str(sum(roc_au)/len(roc_au)))

results_file.close()
