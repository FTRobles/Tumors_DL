# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:06:58 2019

@author: daewo
"""
import os

import numpy as np
import cv2
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard

import gc

#%% Class to define LeNet model architecture, parameters and functions
class LeNet:
    
    def __init__(self):
        self.results_path = os.path.join("..","Resultados","LeNet")
    
    #Extract patches for model training, return the extracted patches and the class that each belong
    def extractPatchesTrain(self,im,mask,size_patch,p = 0.01):
        
        center_patch = round(size_patch/2)
        
        #path image with zeros to extract patches from the image borders
        im2 = np.pad(im,center_patch, mode = 'constant', constant_values = 0)
        
        #Get pixels in the image pixels that belong to tumor (val = 1) and background (val = 0) 
        tumor_idx = np.where(mask != 0)
        back_idx = np.where(mask == 0)
        
        #extract number of patches to extract acoording to the percentage sert by p
        n_tumor = np.shape(tumor_idx)[1]
        n_back = np.shape(back_idx)[1]
        n_samples = round(n_tumor*p)
        
        #get random indexes to tumor and background pixels
        rand_tumor = np.random.randint(0,n_tumor,n_samples)
        rand_back = np.random.randint(0,n_back,n_samples)
        
        patches_img = []
        classes = []
        
        #Get tumor and background patches and their classes
        for pix_tumor, pix_back in zip(rand_tumor, rand_back):
            
            #Get pixel location in padded image
            #Tumor pixel
            x_tumor = tumor_idx[1][pix_tumor] + center_patch
            y_tumor = tumor_idx[0][pix_tumor] + center_patch
            
            #Background pixel
            x_back = back_idx[1][pix_back] + center_patch
            y_back = back_idx[0][pix_back] + center_patch
            
            #Get window limits to extract patch
            tumor_patch_limits = [x_tumor - center_patch, x_tumor + center_patch, y_tumor - center_patch, y_tumor + center_patch]
            back_patch_limits = [x_back - center_patch, x_back + center_patch, y_back - center_patch, y_back + center_patch]
        
            #Extract patches from padded image and class
            patches_img.append(im2[tumor_patch_limits[2]:tumor_patch_limits[3],tumor_patch_limits[0]:tumor_patch_limits[1]])
            classes.append(1)
            patches_img.append(im2[back_patch_limits[2]:back_patch_limits[3],back_patch_limits[0]:back_patch_limits[1]])
            classes.append(0)
            
        return np.array(patches_img), np.array(classes)
 
    #Extract patches for Testing, do not require to get class   
    def extractPatchesTest(self,im,size_patch):
        
        center_patch = round(size_patch/2)
        
        #Pad Test image to extract patches from pixels of the image borders
        im2 = np.pad(im,center_patch, mode = 'constant', constant_values = 0)
        
        patches_img = []
    
        #Get patches for each pixel in the image
        for i in range(im.shape[1]):
            for j in range(im.shape[0]): 
            
                x = i + center_patch
                y = j + center_patch
                
                patch_limits = [x - center_patch, x + center_patch, y - center_patch, y + center_patch]      
                patches_img.append(im2[patch_limits[2]:patch_limits[3],patch_limits[0]:patch_limits[1]])
    
        return np.array(patches_img)
    
    #Define the LeNet-5 architecture
    def modelArch(self,size_patch=28,n_classes=2):
        
        #1 Input Layer: size of images
        #2 Conv Layer: 6 filters of size [5x5] with ReLU
        #3 Pool Layer: Stride = 2 and a 2x2 pooling
        #4 Conv Layer: 16 filters of size [5x5] with ReLU
        #5 Pool Layer: Stride = 2 and a 2x2 pooling
        #6 FC Layer: 120 neurons 
        #7 FC Layer: 84 neurons with dropout regularization rate of 0.4 
        #8 Logits Layer: a FC layer with 10 neurons, one foe each digit target class (0-9) or softmax
        
        #activation = tf.nn.sigmoid
        activation = tf.nn.relu
        
        #1 Input Layer: size of images
        if K.image_data_format() == 'channels_first':
            input_shape = (1, size_patch, size_patch)
        else:
            input_shape = (size_patch, size_patch, 1)
        
        #definition of CONV/POOL layers
        model = tf.keras.Sequential()
        
        #2 Conv Layer: 6 filters of size [5x5] with ReLU
        model.add(tf.keras.layers.Conv2D(
                filters = 6,
                kernel_size = 5,
                padding = "valid",
                strides = 1,
                activation = activation,
                input_shape = input_shape))
        
        #3 Pool Layer: Stride = 2 and a 2x2 pooling
        model.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
        
        #4 Conv Layer: 16 filters of size [5x5] with ReLU
        model.add(tf.keras.layers.Conv2D(
                filters = 16,
                kernel_size = 5,
                padding = "valid",
                strides = 1,
                activation = activation))
        
        #5 Pool Layer: Stride = 2 and a 2x2 pooling
        model.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
        
        #Definition of FC layers
        model.add(tf.keras.layers.Flatten())
        
        #6 FC Layer: 120 neurons 
        model.add(tf.keras.layers.Dense(
                units = 120, 
                activation = activation))
        #7 FC Layer: 84 neurons with dropout regularization rate of 0.4 
        model.add(tf.keras.layers.Dense(
                units = 84, 
                activation = activation))
        #model.add(tf.keras.layers.Dropout(rate = 0.4))  #If training drop out a random percent of the activations
        
        #8 Logits Layer: a FC layer with 10 neurons, one for each class (0 = background || 1 = tumor) or softmax
        model.add(tf.keras.layers.Dense(
                units = n_classes, 
                activation = tf.nn.softmax))
        
        return model

    #Define the model compile options
    def compileModel(self,model):
        
        #Choose the type of tryning algorithm to optimize weights
        #optimizer =  tf.keras.optimizers.RMSprop(lr = 0.01)
        optimizer = "adam"
        
        #Choos the loss function to used to optimize weights
        #loss = root_mean_squared_error
        loss = "categorical_crossentropy"
        
        #Choose metrics to compute loss function
        metrics = ["accuracy"]
        
        #Set compiliation parameters to the model
        model.compile(   
            optimizer = optimizer,
            loss = loss,
            metrics = metrics)
    
        return model
    
    #Train the defined model
    def trainModel(self,train_images,train_masks,model,size_patch=28,n_classes=2, epochs=5, fold=0):
        
        #Extract patch data from training images
        data = np.zeros((1,size_patch,size_patch))
        classes = []
        
        for i in range(np.size(train_images)):
            [patches_img, labels] = self.extractPatchesTrain(train_images[i], train_masks[i], size_patch)
            data = np.concatenate((data,patches_img),axis=0)
            classes = np.concatenate((classes,labels))
    
        #Formating data into nedded model shapes
        data = np.delete(data,0,axis=0)
        
        if K.image_data_format() == 'channels_first':
            data = data.reshape(data.shape[0], 1, size_patch, size_patch)
        else:
            data = data.reshape(data.shape[0], size_patch, size_patch, 1)
        
        # convert class vectors to binary class matrices
        classes = tf.keras.utils.to_categorical(classes, n_classes)
        
        #Shuffle data for model training
        shuffled_indices = np.arange(data.shape[0])
        np.random.shuffle(shuffled_indices)
        shuffled_data = data[shuffled_indices]
        shuffled_classes = classes[shuffled_indices]
        
        #Split data in train and validation sets [90,10]
        samples_count = shuffled_data.shape[0]
        train_samples_count = int(0.9 * samples_count)
        validation_samples_count = int(0.1 * samples_count)
    
        train_data = shuffled_data[:train_samples_count]
        train_classes = shuffled_classes[:train_samples_count]
        validation_data = shuffled_data[train_samples_count:train_samples_count+validation_samples_count]
        validation_classes = shuffled_classes[train_samples_count:train_samples_count+validation_samples_count]
    
        #defining training paramenters
        #Number of evaluation images used in each batch
        batch_size = round(train_data.shape[0]*0.1)
        
        #Stop when the loss grows instead of diminish
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=5) #To check overfitting
        
        #Tensorboard 
        log_dir = os.path.join(self.results_path,"logs","fit",datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = TensorBoard(log_dir=log_dir)
        
        # Create a callback that saves the model's weights
#        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=train_dir,
#                                                         save_weights_only=True,
#                                                         verbose=1)
        
        #Train the model
        model.fit(train_data,train_classes,
                             batch_size = batch_size,
                             epochs = epochs,
                             callbacks=[early_stopping,tensorboard_callback],
                             validation_data = (validation_data,validation_classes),
                             verbose = 1)
        
        train_file = "LeNet_Weights_fold_" + str(fold) + ".h5"
        train_dir = os.path.join(self.results_path,"train_weights",train_file)
        model.save_weights(train_dir)
        
        return model
    
    # Predict class with model 
    def predictClassModel(self,images,model,size_patch=28):
    
        prob_images = []
        class_images = []
        
        for image in images:
        
            data = self.extractPatchesTest(image,size_patch)
            
            if K.image_data_format() == 'channels_first':
                data = data.reshape(data.shape[0], 1, size_patch, size_patch)
            else:
                data = data.reshape(data.shape[0], size_patch, size_patch, 1)
                
            predictions = model.predict(data)
            
            tumor_prob = predictions[:,1]
            classes = predictions.argmax(axis=-1)
            
            prob_image = tumor_prob.reshape(image.shape,order = 'F')
            class_image = classes.reshape(image.shape,order = 'F')
            
            prob_images.append(prob_image)
            class_images.append(class_image)
    
        return prob_images, class_images
        
    
#%% Class to read images. It can be altered to read images while training and not before
class DataGen(keras.utils.Sequence):
    def __init__(self, ids, path, image_size=128):
        self.ids = ids
        self.path = path
        self.image_size = image_size
        
    #load an image and its mask    
    def __load__(self, id_name):
        
        ## Path
        image_path = os.path.join(self.path,id_name)
        mask_path = os.path.join(self.path,"Mask/",id_name)
        
        ## Reading Image
        image = cv2.imread(image_path, 0)
        
        ## Reading Mask
        mask = cv2.imread(mask_path, 0)
            
        ## Normalizaing 
        image = image/255.0
        mask = mask/255.0
        
        return image, mask
    
    def __load_all__(self):
        
        #lists that will store the images
        tumor_images = []
        mask_images = []

        #Read all images and its masks
        for i in range(len(self.ids)):
        
            x, y = self.__load__(self.ids[i])
            tumor_images.append(x)
            mask_images.append(y)
        
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