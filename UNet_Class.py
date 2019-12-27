# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:06:58 2019

@author: daewo
"""
import os
from datetime import datetime
import gc
import imageio
from random import randint

import numpy as np

import cv2

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
#from tensorflow.keras.preprocessing.image import ImageDataGenerator 

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapOnImage

#%% Class to define UNet model architecture, parameters and functions
class UNet:
    
    def __init__(self):
        self.results_path = os.path.join("..","Resultados","UNet")
        
    def down_block(self,x, filters, kernel_size=(3, 3), padding="same", strides=1):
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
        return c, p
    
    def up_block(self,x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
        us = keras.layers.UpSampling2D((2, 2))(x)
        concat = keras.layers.Concatenate()([us, skip])
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        return c
    
    def bottleneck(self,x, filters, kernel_size=(3, 3), padding="same", strides=1):
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        return c
        
    #Define the LeNet-5 architecture
    def modelArch(self,image_size=128):
        
        f = [16, 32, 64, 128, 256]
        inputs = keras.layers.Input((image_size, image_size, 3))
        
        p0 = inputs
        c1, p1 = self.down_block(p0, f[0]) #128 -> 64
        c2, p2 = self.down_block(p1, f[1]) #64 -> 32
        c3, p3 = self.down_block(p2, f[2]) #32 -> 16
        c4, p4 = self.down_block(p3, f[3]) #16->8
        
        bn = self.bottleneck(p4, f[4])
        
        u1 = self.up_block(bn, c4, f[3]) #8 -> 16
        u2 = self.up_block(u1, c3, f[2]) #16 -> 32
        u3 = self.up_block(u2, c2, f[1]) #32 -> 64
        u4 = self.up_block(u3, c1, f[0]) #64 -> 128
        
        outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
        model = keras.models.Model(inputs, outputs)
        
        return model

    #Define the model compile options
    def compileModel(self,model):
        
        #Choose the type of tryning algorithm to optimize weights
        #optimizer =  tf.keras.optimizers.RMSprop(lr = 0.01)
        optimizer = "adam"
        
        #Choos the loss function to used to optimize weights
        loss = "binary_crossentropy"
        
        #Choose metrics to compute loss function
        metrics = ["accuracy"]
        
        #Set compiliation parameters to the model
        model.compile(   
            optimizer = optimizer,
            loss = loss,
            metrics = metrics)

        return model
    
    #Train the defined model
    def trainModel(self,train_images,train_masks,model,image_size=128,epochs=30,fold=0):
        
        shuffled_indices = np.arange(train_images.shape[0])
        np.random.shuffle(shuffled_indices)
        shuffled_tumors = train_images[shuffled_indices]
        shuffled_masks = train_masks[shuffled_indices]
        
        #slpit data in train, validation and test sets [90,10]
        samples_count = shuffled_tumors.shape[0]
        train_samples_count = int(0.9 * samples_count)
        validation_samples_count = int(0.1 * samples_count)
    
        train_images = shuffled_tumors[:train_samples_count]
        train_masks = shuffled_masks[:train_samples_count]
        validation_tumors = shuffled_tumors[train_samples_count:train_samples_count+validation_samples_count]
        validation_masks = shuffled_masks[train_samples_count:train_samples_count+validation_samples_count]
        
        # Define our augmentation pipeline.
        seq = iaa.Sequential([
#            iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
#            iaa.Sharpen((0.0, 1.0)),       # sharpen the image
            iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects segmaps)
            iaa.Affine(shear=(-45, 45)),
#            iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects segmaps)
        ], random_order=True)
    
        # Augment images and segmaps.
#        train_images = (train_images*255).astype(np.uint8)
#        train_masks = train_masks.astype(np.uint8)
        
        images_aug = train_images
        masks_aug = train_masks
        
        for i in range(5):
            images_aug_i, masks_aug_i = seq(images = images_aug, segmentation_maps = masks_aug)
            train_images = np.concatenate((train_images,images_aug_i),axis=0)
            train_masks = np.concatenate((train_masks,masks_aug_i),axis=0)   
        
        #defining training paramenters
        #Number of evaluation images used in each batch
        batch_size = round(train_images.shape[0]*0.1)
        
        #Stop when the loss grows instead of diminish
#        early_stopping = tf.keras.callbacks.EarlyStopping(patience=5) #To check overfitting
        
        #Tensorboard 
        logdir = os.path.join("..","Resultados","UNet","logs","fit",datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = TensorBoard(log_dir=logdir)
        
        ## Normalizaing 
        train_images = train_images/255.0
        train_masks = train_masks/255.0
        
#        #Train the model
        print("Training fold: " + str(fold))   
        model.fit(train_images, train_masks,
                             batch_size = batch_size,
                             epochs = epochs,
                             callbacks=[tensorboard_callback],
                             validation_data = (validation_tumors,validation_masks),
                             verbose = 1)        
        
        train_file = "UNet_Weights_fold_" + str(fold) + ".h5"
        train_dir = os.path.join(self.results_path,"train_weights",train_file)
        model.save_weights(train_dir)
        
        return model
    
    # Predict class with model 
    def predictClassModel(self,images,model,image_size=128):
    
        prob_images = []
        class_images = []
        
        ## Normalizaing 
        images = images/255
        
        for image in images:
            
            ## Dataset for prediction
            test_image = image.reshape(1, image_size, image_size, 3)
            
            prob_image = model.predict(test_image)
            class_image = prob_image > 0.5
            
            class_image = class_image[0][:,:,0]
            prob_image = prob_image[0][:,:,0]
            
            prob_images.append(prob_image)
            class_images.append(class_image)
    
        return prob_images, class_images
    
#%% Class to read images. It can be altered to read images while training and not before
class DataGen(keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=8, image_size=128):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()
        
    def __load__(self, id_name):
        
        ## Path
        image_path = os.path.join(self.path,id_name)
        mask_path = os.path.join(self.path,"Mask/",id_name)
        
        ## Reading Image
        image = cv2.imread(image_path, 0)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        
        ## Reading Mask
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size),interpolation=cv2.INTER_CUBIC)
        mask = np.expand_dims(mask, axis=-1)
            
        ## Normalizaing 
#        image = image/255.0
#        mask = mask/255.0
        
        return image.astype(np.uint8), mask.astype(np.uint8)
    
    def __load_all__(self):
        
        #lists that will store the images
        tumor_images = []
        mask_images = []

        #Read all images and its masks
        for i in range(len(self.ids)):
        
            x, y = self.__load__(self.ids[i])
            tumor_images.append(x)
            mask_images.append(y)
    
        #Formating data into nedded model shapes
        tumor_images = np.array(tumor_images)
        mask_images = np.array(mask_images)
        
        return tumor_images, mask_images
    
#%% Free memory after predicting

# Reset Keras Session
def ResetKeras(model):
    sess = tf.compat.v1.keras.backend.get_session()
    tf.compat.v1.keras.backend.clear_session()
    sess.close()
    sess = tf.compat.v1.keras.backend.get_session()

    try:
        del model # this is from global space - change this as you need
    except:
        pass

    print(gc.collect()) # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))