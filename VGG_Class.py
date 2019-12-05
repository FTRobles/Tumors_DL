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
from tensorflow.keras.callbacks import TensorBoard

import gc

#%% Class to define UNet model architecture, parameters and functions
class VGG:
    
    def __init__(self):
        self.results_path = os.path.join("..","Resultados","FCN_VGG")
        
    #Define the VGG16 with 8 FC layers
    def modelArch(self,n_classes=2, VGG_Weights_path = "../Datos/weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5", image_size = 128):
        
        ## input_height and width must be devisible by 32 because maxpooling with filter size = (2,2) is operated 5 times,
        ## which makes the input_height and width 2^5 = 32 times smaller
        assert image_size%32 == 0
        IMAGE_ORDERING =  "channels_last" 
    
        img_input = keras.layers.Input(shape=(image_size,image_size, 3)) ## Assume 224,224,3
        
        ## Block 1
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING )(img_input)
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING )(x)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
        f1 = x
        
        # Block 2
        x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(x)
        x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(x)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)
        f2 = x
    
        # Block 3
        x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING )(x)
        x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING )(x)
        x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING )(x)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)
        pool3 = x
    
        # Block 4
        x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING )(x)
        x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING )(x)
        x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING )(x)
        pool4 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(x)## (None, 14, 14, 512) 
    
        # Block 5
        x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING )(pool4)
        x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING )(x)
        x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING )(x)
        pool5 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING )(x)## (None, 7, 7, 512)
    
        #Fully Conected layers of VGG
        #x = Flatten(name='flatten')(x)
        #x = Dense(4096, activation='relu', name='fc1')(x)
        # <--> o = ( Conv2D( 4096 , ( 7 , 7 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)
        # assuming that the input_height = input_width = 224 as in VGG data
        
        #x = Dense(4096, activation='relu', name='fc2')(x)
        # <--> o = ( Conv2D( 4096 , ( 1 , 1 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)   
        # assuming that the input_height = input_width = 224 as in VGG data
        
        #x = Dense(1000 , activation='softmax', name='predictions')(x)
        # <--> o = ( Conv2D( nClasses ,  ( 1 , 1 ) ,kernel_initializer='he_normal' , data_format=IMAGE_ORDERING))(o)
        # assuming that the input_height = input_width = 224 as in VGG data
        
        
        vgg  = keras.models.Model(  img_input , pool5  )
        vgg.load_weights(VGG_Weights_path) ## loading VGG weights for the encoder parts of FCN8
        
        n = 4096
        o = ( keras.layers.Conv2D( n , ( 7 , 7 ) , activation='relu' , padding='same', name="conv6", data_format=IMAGE_ORDERING))(pool5)
        conv7 = ( keras.layers.Conv2D( n , ( 1 , 1 ) , activation='relu' , padding='same', name="conv7", data_format=IMAGE_ORDERING))(o)
        
        
        ## 4 times upsamping for pool4 layer
        conv7_4 = keras.layers.Conv2DTranspose( n_classes , kernel_size=(4,4) ,  strides=(4,4) , use_bias=False, data_format=IMAGE_ORDERING )(conv7)
        ## (None, 224, 224, 10)
        ## 2 times upsampling for pool411
        pool411 = ( keras.layers.Conv2D( n_classes , ( 1 , 1 ) , activation='relu' , padding='same', name="pool4_11", data_format=IMAGE_ORDERING))(pool4)
        pool411_2 = (keras.layers.Conv2DTranspose( n_classes , kernel_size=(2,2) ,  strides=(2,2) , use_bias=False, data_format=IMAGE_ORDERING ))(pool411)
        
        pool311 = ( keras.layers.Conv2D( n_classes , ( 1 , 1 ) , activation='relu' , padding='same', name="pool3_11", data_format=IMAGE_ORDERING))(pool3)
            
        o = keras.layers.Add(name="add")([pool411_2, pool311, conv7_4 ])
        o = keras.layers.Conv2DTranspose( n_classes , kernel_size=(8,8) ,  strides=(8,8) , use_bias=False, data_format=IMAGE_ORDERING )(o)
        o = (keras.layers.Activation('softmax'))(o)
        
        model = keras.models.Model(img_input, o)
    
        return model

    #Define the model compile options
    def compileModel(self,model):
        
        #Choose the type of tryning algorithm to optimize weights
        #optimizer =  tf.keras.optimizers.RMSprop(lr = 0.01)
        optimizer = tf.keras.optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
        
        #Choos the loss function to used to optimize weights
        loss = 'categorical_crossentropy'
        
        #Choose metrics to compute loss function
        metrics = ["accuracy"]
            
        #Set compiliation parameters to the model
        model.compile(   
            optimizer = optimizer,
            loss = loss,
            metrics = metrics)
        
        return model
    
    #Train the defined model
    def trainModel(self,train_images,train_masks,model,image_size=128,fold=0):
        
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
        
        
        #defining training paramenters
        #Number of evaluation images used in each batch
        batch_size = round(train_images.shape[0]*0.1)
        
        #Number of epochs to train
        epochs = 30
        
        #Stop when the loss grows instead of diminish
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=5) #To check overfitting
        
        #Tensorboard 
        logdir = os.path.join("..","Resultados","FCN_VGG","logs","fit",datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = TensorBoard(log_dir=logdir)
        
        #Train the model
        model.fit(train_images,train_masks,
                             batch_size = batch_size,
                             epochs = epochs,
                             callbacks=[early_stopping,tensorboard_callback],
                             validation_data = (validation_tumors,validation_masks),
                             verbose = 1)
    
        train_file = "VGG_Weights_fold_" + str(fold) + ".h5"
        train_dir = os.path.join(self.results_path,"train_weights",train_file)
        model.save_weights(train_dir)
        
        return model
    
    # Predict class with model 
    def predictClassModel(self,images,model,image_size=128):
    
        prob_images = []
        class_images = []
        
        for image in images:
            
            ## Dataset for prediction
            test_image = image.reshape(1, image_size, image_size, 3)
            
            prob_predicts = model.predict(test_image)
    
            class_image = np.argmax(prob_predicts, axis=3)
            prob_image = prob_predicts[0,:,:,1]
            
            class_image = np.reshape(class_image[0], (image_size, image_size))
            
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
        
    def __load__(self, id_name, n_classes = 2):
        
        ## Path
        image_path = os.path.join(self.path,id_name)
        mask_path = os.path.join(self.path,"Mask/",id_name)
        
        ## Reading Image
        image = cv2.imread(image_path, 1)
        image = np.float32(cv2.resize(image, ( self.image_size , self.image_size ))) / 127.5
#        image = cv2.imread(image_path, 0)
#        image = cv2.resize(image, (self.image_size, self.image_size))
#        image = np.float32(cv2.cvtColor(image,cv2.COLOR_GRAY2RGB))

        
        ## Reading Mask
        seg_labels = np.zeros((  self.image_size , self.image_size  , n_classes ))
        mask = cv2.imread(mask_path, 1)/255
        mask = cv2.resize(mask, (self.image_size , self.image_size ))
        mask = mask[:, : , 0]

        for c in range(n_classes):
            seg_labels[: , : , c ] = (mask == c ).astype(int)

        
        return image, seg_labels
    
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