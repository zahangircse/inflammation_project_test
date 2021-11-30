#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 22:25:17 2018
# Reference:
Alom, Md Zahangir, et al. "Recurrent residual convolutional neural network based on u-net (r2u-net) for medical image segmentation." arXiv preprint arXiv:1802.06955 (2018).
@author: zahangir
"""
from __future__ import print_function
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
# import tf.keras.backend as K
# K.image_dim_ordering() == 'tf'

# smooth = 1.

# # Metric function
# def dice_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# # Loss funtion
# def dice_coef_loss(y_true, y_pred):
#     return -dice_coef(y_true, y_pred)



def Rec_conv2d_bn(x, nb_filter, nb_row, nb_col,border_mode='same',strides=(1, 1),name=None):
    '''Utility function to apply conv + BN.
    '''
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
        
    #if K.image_dim_ordering() == 'tf':
    bn_axis = 3
    
    x1 = tf.keras.layers.Conv2D(nb_filter, (nb_row, nb_col),strides=strides,
                      padding=border_mode,
                      name=conv_name,kernel_regularizer=tf.keras.regularizers.l2(0.0002),data_format='channels_last')(x)
    x1=tf.keras.layers.Activation('relu')(x1)                  
    
    x2 = tf.keras.layers.Conv2D(nb_filter, (nb_row, nb_col),strides=strides,
                      padding=border_mode,
                      name=conv_name,kernel_regularizer=tf.keras.regularizers.l2(0.0002),data_format='channels_last')(x1)              
    x2=tf.keras.layers.Activation('relu')(x2)                   
    x12 = tf.keras.layers.add([x1, x2])
    
    x3 = tf.keras.layers.Conv2D(nb_filter, (nb_row, nb_col),strides=strides,
                      padding=border_mode,
                      name=conv_name,kernel_regularizer=tf.keras.regularizers.l2(0.0002),data_format='channels_last')(x12) 
    x3=tf.keras.layers.Activation('relu')(x3)      
    x13 = tf.keras.layers.add([x1, x3])

    x4 = tf.keras.layers.Conv2D(nb_filter, (nb_row, nb_col),strides=strides,
                      padding='same',
                      name=conv_name,kernel_regularizer=tf.keras.regularizers.l2(0.0002),data_format='channels_last')(x13)
    x4=tf.keras.layers.Activation('relu')(x4) 
   
    x14 = tf.keras.layers.add([x1, x4])
    
    x=tf.keras.layers.Activation('relu')(x14)
    
    #x = BatchNormalization(axis=bn_axis, name=bn_name)(x13)
    return x

#Define the neural network

def build_R2UNetE(input_shape,num_classes):
    
      
    inputs = tf.keras.Input(input_shape)   
    channel_axis = 3
    #pdb.set_trace()
    # first block
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(inputs)    
    rcnn_bn1 = Rec_conv2d_bn(x, 32, 3, 3)    
    conv1_f = tf.keras.layers.add([x, rcnn_bn1])

    # downsampling first block..
    pool1 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv1_f)
    conv_pool1 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool1)
    
    # second RRCNN layer...
    rcnn_bn2 = Rec_conv2d_bn(conv_pool1, 64, 3, 3)    
    conv2_f = tf.keras.layers.add([conv_pool1, rcnn_bn2])
    
    # downsampling first block..
    pool2 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv2_f)
    conv_pool2 = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool2)
    
    # third RRCNN layer...
    rcnn_bn3 = Rec_conv2d_bn(conv_pool2, 128, 3, 3)    
    conv3_f = tf.keras.layers.add([conv_pool2, rcnn_bn3])
    
    # downsampling first block..
    pool3 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv3_f)
    conv_pool3 = tf.keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool3)
    
    # fourth RRCNN layer...
    rcnn_bn4 = Rec_conv2d_bn(conv_pool3, 256, 3, 3)    
    conv4_f = tf.keras.layers.add([conv_pool3, rcnn_bn4])
      
    # Decoder...
   
    #pdb.set_trace()
    
    up7_1 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv4_f)
    up7 = tf.keras.layers.concatenate([up7_1, conv3_f], axis=channel_axis)
    conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(up7)
    conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv7)
    
    up8_1 =tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv7)
    up8 = tf.keras.layers.concatenate([up8_1, conv2_f], axis=channel_axis)
    conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(up8)
    conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv8)
    
    up9_1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv8)
    up9 = tf.keras.layers.concatenate([up9_1, conv1_f], axis=channel_axis)
    conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(up9)
    conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv9)
    
    if num_classes <= 2:
        conv10 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid',data_format='channels_last')(conv9)
    else:
        conv10 = tf.keras.layers.Conv2D(num_classes,(1,1), activation='softmax')(conv9)

    model = tf.keras.Model(inputs=[inputs], outputs=[conv10])

    #model.compile(optimizer=Adam(2e-4),loss=dice_coef_loss,metrics = [dice_coef, 'acc', 'mse'])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss='binary_crossentropy',metrics = ['acc', 'mse'])

    
    return model 


def build_R2UNetED(input_shape,num_classes):
      

   channel_axis = 3
   inputs =  tf.keras.Input(input_shape) 

   # zero block
   x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same',data_format='channels_last')(inputs)    
   rcnn_bn0 = Rec_conv2d_bn(x, 16, 3, 3)    
   conv0_f = tf.keras.layers.add([x, rcnn_bn0])

    # downsampling zero block..
   pool0 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv0_f)
   conv_pool0 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool0)

    # first block
   #x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(inputs)    
   rcnn_bn1 = Rec_conv2d_bn(conv_pool0, 32, 3, 3)    
   conv1_f = tf.keras.layers.add([conv_pool0, rcnn_bn1])

    # downsampling first block..
   pool1 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv1_f)
   conv_pool1 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool1)
    
    # second RRCNN layer...
   rcnn_bn2 = Rec_conv2d_bn(conv_pool1, 64, 3, 3)    
   conv2_f = tf.keras.layers.add([conv_pool1, rcnn_bn2])
    
    # downsampling first block..
   pool2 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv2_f)
   conv_pool2 = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool2)
    
    # third RRCNN layer...
   rcnn_bn3 = Rec_conv2d_bn(conv_pool2, 128, 3, 3)    
   conv3_f = tf.keras.layers.add([conv_pool2, rcnn_bn3])
    
    # downsampling first block..
   pool3 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv3_f)
   conv_pool3 = tf.keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool3)
    
    # fourth RRCNN layer...
   rcnn_bn4 = Rec_conv2d_bn(conv_pool3, 256, 3, 3)    
   conv4_f = tf.keras.layers.add([conv_pool3, rcnn_bn4])
   
   # fifth block .....
   pool4 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv4_f)
   conv_pool4 = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool4)
    
    # fourth RRCNN layer...
   rcnn_bn5 = Rec_conv2d_bn(conv_pool4, 512, 3, 3)    
   conv5_f = tf.keras.layers.add([conv_pool4, rcnn_bn5])
   
   
   
    # Decoder...
   up6_1 = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv5_f)
   up6 = tf.keras.layers.concatenate([up6_1, conv4_f], axis=channel_axis)
   up6 = Rec_conv2d_bn(up6,256, 3, 3)
   conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same',data_format='channels_last')(up6)
   

   up7_1 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv6)
   up7 = tf.keras.layers.concatenate([up7_1, conv3_f], axis=channel_axis)
   up7 = Rec_conv2d_bn(up7,128, 3, 3)
   conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(up7)
    #conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv7)
    
   up8_1 =tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv7)
   up8 = tf.keras.layers.concatenate([up8_1, conv2_f], axis=channel_axis)
   up8 = Rec_conv2d_bn(up8,64, 3, 3)
   conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(up8)
    #conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv8)
    
   up9_1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv8)
   up9 = tf.keras.layers.concatenate([up9_1, conv1_f], axis=channel_axis)
   up9 = Rec_conv2d_bn(up9,32, 3, 3)
   conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(up9)
    #conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv9)
   
   up10_1 = tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv9)
   up10 = tf.keras.layers.concatenate([up10_1, conv0_f], axis=channel_axis)
   up10 = Rec_conv2d_bn(up10,16, 3, 3)
   conv10 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same',data_format='channels_last')(up10)
   
   if num_classes <= 2:
        conv10 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid',data_format='channels_last')(conv10)
   else:
        conv10 = tf.keras.layers.Conv2D(num_classes,(1,1), activation='softmax')(conv10)
       

   #model = tf.keras.Model(inputs=[inputs], outputs=[conv10])
   model = tf.keras.Model(inputs=inputs, outputs=conv10)


   #model.compile(optimizer=Adam(2e-4),loss=dice_coef_loss,metrics = [dice_coef, 'acc', 'mse'])
    
   model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss='binary_crossentropy',metrics = ['acc', 'mse'])
    
   return model 

def build_R2UNetED_final(input_shape,num_classes):
      

   channel_axis = 3
   inputs =  tf.keras.Input(input_shape) 

   # zero block
   x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same',data_format='channels_last')(inputs)    
   rcnn_bn0 = Rec_conv2d_bn(x, 16, 3, 3)    
   conv0_f = tf.keras.layers.add([x, rcnn_bn0])

    # downsampling zero block..
   pool0 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv0_f)
   conv_pool0 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool0)

    # first block
   #x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(inputs)    
   rcnn_bn1 = Rec_conv2d_bn(conv_pool0, 32, 3, 3)    
   conv1_f = tf.keras.layers.add([conv_pool0, rcnn_bn1])

    # downsampling first block..
   pool1 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv1_f)
   conv_pool1 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool1)
    
    # second RRCNN layer...
   rcnn_bn2 = Rec_conv2d_bn(conv_pool1, 64, 3, 3)    
   conv2_f = tf.keras.layers.add([conv_pool1, rcnn_bn2])
    
    # downsampling first block..
   pool2 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv2_f)
   conv_pool2 = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool2)
    
    # third RRCNN layer...
   rcnn_bn3 = Rec_conv2d_bn(conv_pool2, 128, 3, 3)    
   conv3_f = tf.keras.layers.add([conv_pool2, rcnn_bn3])
    
    # downsampling first block..
   pool3 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv3_f)
   conv_pool3 = tf.keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool3)
    
    # fourth RRCNN layer...
   rcnn_bn4 = Rec_conv2d_bn(conv_pool3, 256, 3, 3)    
   conv4_f = tf.keras.layers.add([conv_pool3, rcnn_bn4])
   
   # fifth block .....
   pool4 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv4_f)
   conv_pool4 = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool4)
    
    # fourth RRCNN layer...
   rcnn_bn5 = Rec_conv2d_bn(conv_pool4, 512, 3, 3)    
   conv5_f = tf.keras.layers.add([conv_pool4, rcnn_bn5])
     
    # Decoder...
   up6_1 = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv5_f)
   up6_concat = tf.keras.layers.concatenate([up6_1, conv4_f], axis=channel_axis)
   up6_concat = tf.keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same',data_format='channels_last')(up6_concat)
   up6_recconv = Rec_conv2d_bn(up6_concat,256, 3, 3)
   up6_f = tf.keras.layers.add([up6_recconv, up6_concat])
   conv6 = tf.keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same',data_format='channels_last')(up6_f)
   
   up7_1 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv6)
   up7_concat = tf.keras.layers.concatenate([up7_1, conv3_f], axis=channel_axis)
   up7_concat = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(up7_concat)
   up7_recconv = Rec_conv2d_bn(up7_concat,128, 3, 3)
   up7_f = tf.keras.layers.add([up7_recconv, up7_concat])
   conv7 = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(up7_f)

   up8_1 =tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv7)
   up8_concat = tf.keras.layers.concatenate([up8_1, conv2_f], axis=channel_axis)
   up8_concat = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(up8_concat)
   up8_recconv = Rec_conv2d_bn(up8_concat,64, 3, 3)
   up8_f = tf.keras.layers.add([up8_recconv, up8_concat])
   conv8 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(up8_f)
 
   up9_1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv8)
   up9_concat = tf.keras.layers.concatenate([up9_1, conv1_f], axis=channel_axis)
   up9_concat = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(up9_concat)
   up9_recconv = Rec_conv2d_bn(up9_concat,32, 3, 3)
   up9_f = tf.keras.layers.add([up9_recconv, up9_concat])
   conv9 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(up9_f)

   up10_1 = tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv9)
   up10_concat = tf.keras.layers.concatenate([up10_1, conv0_f], axis=channel_axis)
   up10_concat = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(up10_concat)
   up10_recconv = Rec_conv2d_bn(up10_concat,16, 3, 3)
   up10_f = tf.keras.layers.add([up10_recconv, up10_concat])
   conv10 = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(up10_f)
   
   if num_classes <= 2:
        conv10 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid',data_format='channels_last')(conv10)
   else:
        conv10 = tf.keras.layers.Conv2D(num_classes,(1,1), activation='softmax')(conv10)
       

   #model = tf.keras.Model(inputs=[inputs], outputs=[conv10])
   model = tf.keras.Model(inputs=inputs, outputs=conv10)


   #model.compile(optimizer=Adam(2e-4),loss=dice_coef_loss,metrics = [dice_coef, 'acc', 'mse'])
    
   model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss='binary_crossentropy',metrics = ['acc', 'mse'])
    
   return model 

def build_R2UNetED_DP_final(input_shape,num_classes):
      

   channel_axis = 3
   inputs =  tf.keras.Input(input_shape) 

   # zero block
   x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(inputs)    
   rcnn_bn0 = Rec_conv2d_bn(x, 32, 3, 3)    
   conv0_f = tf.keras.layers.add([x, rcnn_bn0])

    # downsampling zero block..
   pool0 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv0_f)
   conv_pool0 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool0)

    # first block
   #x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(inputs)    
   rcnn_bn1 = Rec_conv2d_bn(conv_pool0, 64, 3, 3)    
   conv1_f = tf.keras.layers.add([conv_pool0, rcnn_bn1])

    # downsampling first block..
   pool1 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv1_f)
   conv_pool1 = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool1)
    
    # second RRCNN layer...
   rcnn_bn2 = Rec_conv2d_bn(conv_pool1, 128, 3, 3)    
   conv2_f = tf.keras.layers.add([conv_pool1, rcnn_bn2])
    
    # downsampling first block..
   pool2 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv2_f)
   conv_pool2 = tf.keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool2)
    
    # third RRCNN layer...
   rcnn_bn3 = Rec_conv2d_bn(conv_pool2, 256, 3, 3)    
   conv3_f = tf.keras.layers.add([conv_pool2, rcnn_bn3])
    
    # downsampling first block..
   pool3 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv3_f)
   conv_pool3 = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool3)
    
    # fourth RRCNN layer...
   rcnn_bn4 = Rec_conv2d_bn(conv_pool3, 512, 3, 3)    
   conv4_f = tf.keras.layers.add([conv_pool3, rcnn_bn4])
   
   # fifth block .....
   pool4 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv4_f)
   conv_pool4 = tf.keras.layers.Conv2D(1024, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool4)
    
    # fourth RRCNN layer...
   rcnn_bn5 = Rec_conv2d_bn(conv_pool4, 1024, 3, 3)    
   conv5_f = tf.keras.layers.add([conv_pool4, rcnn_bn5])
     
    # Decoder...
   up6_1 = tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv5_f)
   up6_concat = tf.keras.layers.concatenate([up6_1, conv4_f], axis=channel_axis)
   up6_concat = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same',data_format='channels_last')(up6_concat)
   up6_recconv = Rec_conv2d_bn(up6_concat,512, 3, 3)
   up6_f = tf.keras.layers.add([up6_recconv, up6_concat])
   conv6 = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same',data_format='channels_last')(up6_f)
   
   up7_1 = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv6)
   up7_concat = tf.keras.layers.concatenate([up7_1, conv3_f], axis=channel_axis)
   up7_concat = tf.keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same',data_format='channels_last')(up7_concat)
   up7_recconv = Rec_conv2d_bn(up7_concat,256, 3, 3)
   up7_f = tf.keras.layers.add([up7_recconv, up7_concat])
   conv7 = tf.keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same',data_format='channels_last')(up7_f)

   up8_1 =tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv7)
   up8_concat = tf.keras.layers.concatenate([up8_1, conv2_f], axis=channel_axis)
   up8_concat = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(up8_concat)
   up8_recconv = Rec_conv2d_bn(up8_concat,128, 3, 3)
   up8_f = tf.keras.layers.add([up8_recconv, up8_concat])
   conv8 = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(up8_f)
 
   up9_1 = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv8)
   up9_concat = tf.keras.layers.concatenate([up9_1, conv1_f], axis=channel_axis)
   up9_concat = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(up9_concat)
   up9_recconv = Rec_conv2d_bn(up9_concat,64, 3, 3)
   up9_f = tf.keras.layers.add([up9_recconv, up9_concat])
   conv9 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(up9_f)

   up10_1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv9)
   up10_concat = tf.keras.layers.concatenate([up10_1, conv0_f], axis=channel_axis)
   up10_concat = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(up10_concat)
   up10_recconv = Rec_conv2d_bn(up10_concat,32, 3, 3)
   up10_f = tf.keras.layers.add([up10_recconv, up10_concat])
   conv10 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(up10_f)
   
   if num_classes <= 2:
        conv10 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid',data_format='channels_last')(conv10)
   else:
        conv10 = tf.keras.layers.Conv2D(num_classes,(1,1), activation='softmax')(conv10)
       

   #model = tf.keras.Model(inputs=[inputs], outputs=[conv10])
   model = tf.keras.Model(inputs=inputs, outputs=conv10)


   #model.compile(optimizer=tf.train.AdamOptimizer(3e-4),loss=dice_coef_loss,metrics = [dice_coef, 'acc', 'mse'])
    
   model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss='binary_crossentropy',metrics = ['acc', 'mse'])
    
   return model 



def build_DeltaNetA_final(input_shape,num_classes):
      

   channel_axis = 3
   inputs =  tf.keras.Input(input_shape) 

   # zero block
   x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same',data_format='channels_last')(inputs)    
   rcnn_bn0 = Rec_conv2d_bn(x, 16, 3, 3)    
   conv0_f = tf.keras.layers.add([x, rcnn_bn0])

    # downsampling zero block..
   pool0 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv0_f)
   conv_pool0 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool0)

    # first block
   #x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(inputs)    
   rcnn_bn1 = Rec_conv2d_bn(conv_pool0, 32, 3, 3)    
   conv1_f = tf.keras.layers.add([conv_pool0, rcnn_bn1])

    # downsampling first block..
   pool1 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv1_f)
   conv_pool1 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool1)
    
    # second RRCNN layer...
   rcnn_bn2 = Rec_conv2d_bn(conv_pool1, 64, 3, 3)    
   conv2_f = tf.keras.layers.add([conv_pool1, rcnn_bn2])
    
    # downsampling first block..
   pool2 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv2_f)
   conv_pool2 = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool2)
    
    # third RRCNN layer...
   rcnn_bn3 = Rec_conv2d_bn(conv_pool2, 128, 3, 3)    
   conv3_f = tf.keras.layers.add([conv_pool2, rcnn_bn3])
    
    # downsampling first block..
   pool3 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv3_f)
   conv_pool3 = tf.keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool3)
    
    # fourth RRCNN layer...
   rcnn_bn4 = Rec_conv2d_bn(conv_pool3, 256, 3, 3)    
   conv4_f = tf.keras.layers.add([conv_pool3, rcnn_bn4])
   
   # fifth block .....
   pool4 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv4_f)
   conv_pool4 = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool4)
    
    # fourth RRCNN layer...
   rcnn_bn5 = Rec_conv2d_bn(conv_pool4, 512, 3, 3)    
   conv5_f = tf.keras.layers.add([conv_pool4, rcnn_bn5])
     
    # Decoding from first laten space.
    
   # Decoding from seocond latent space....
   '''
   # Decoding from fourth laten space.
   ts4_up9_1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv2_f)
   ts4_up9_concat = tf.keras.layers.concatenate([ts4_up9_1, conv1_f], axis=channel_axis)
   ts4_up9_concat = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts4_up9_concat)
   ts4_up9_recconv = Rec_conv2d_bn(ts4_up9_concat,32, 3, 3)
   ts4_up9_f = tf.keras.layers.add([ts4_up9_recconv, ts4_up9_concat])
   ts4_conv9 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts4_up9_f)

   ts4_up10_1 = tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts4_conv9)
   ts4_up10_concat = tf.keras.layers.concatenate([ts4_up10_1, conv0_f], axis=channel_axis)
   ts4_up10_concat = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts4_up10_concat)
   ts4_up10_recconv = Rec_conv2d_bn(ts4_up10_concat,16, 3, 3)
   ts4_up10_f = tf.keras.layers.add([ts4_up10_recconv, ts4_up10_concat])
   ts4_conv10 = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts4_up10_f)
   

    
   # Decoding from third laten space.
   ts3_up8_1 =tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv3_f)
   ts3_up8_concat = tf.keras.layers.concatenate([ts3_up8_1, conv2_f], axis=channel_axis)
   ts3_up8_concat = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts3_up8_concat)
   ts3_up8_recconv = Rec_conv2d_bn(ts3_up8_concat,64, 3, 3)
   ts3_up8_f = tf.keras.layers.add([ts3_up8_recconv, ts3_up8_concat])
   ts3_conv8 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts3_up8_f)
 
   ts3_up9_1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts3_conv8)
   ts3_up9_concat = tf.keras.layers.concatenate([ts3_up9_1, conv1_f], axis=channel_axis)
   ts3_up9_concat = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts3_up9_concat)
   ts3_up9_recconv = Rec_conv2d_bn(ts3_up9_concat,32, 3, 3)
   ts3_up9_f = tf.keras.layers.add([ts3_up9_recconv, ts3_up9_concat])
   ts3_conv9 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts3_up9_f)

   ts3_up10_1 = tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts3_conv9)
   ts3_up10_concat = tf.keras.layers.concatenate([ts3_up10_1, conv0_f], axis=channel_axis)
   ts3_up10_concat = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts3_up10_concat)
   ts3_up10_recconv = Rec_conv2d_bn(ts3_up10_concat,16, 3, 3)
   ts3_up10_f = tf.keras.layers.add([ts3_up10_recconv, ts3_up10_concat])
   ts3_conv10 = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts3_up10_f)
      
   '''
   # Decoding from fourth laten space.
   
   ts2_up7_1 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv4_f)
   ts2_up7_concat = tf.keras.layers.concatenate([ts2_up7_1, conv3_f], axis=channel_axis)
   ts2_up7_concat = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up7_concat)
   ts2_up7_recconv = Rec_conv2d_bn(ts2_up7_concat,128, 3, 3)
   ts2_up7_f = tf.keras.layers.add([ts2_up7_recconv, ts2_up7_concat])
   ts2_conv7 = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up7_f)

   ts2_up8_1 =tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts2_conv7)
   ts2_up8_concat = tf.keras.layers.concatenate([ts2_up8_1, conv2_f], axis=channel_axis)
   ts2_up8_concat = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up8_concat)
   ts2_up8_recconv = Rec_conv2d_bn(ts2_up8_concat,64, 3, 3)
   ts2_up8_f = tf.keras.layers.add([ts2_up8_recconv, ts2_up8_concat])
   ts2_conv8 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up8_f)
 
   ts2_up9_1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts2_conv8)
   ts2_up9_concat = tf.keras.layers.concatenate([ts2_up9_1, conv1_f], axis=channel_axis)
   ts2_up9_concat = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up9_concat)
   ts2_up9_recconv = Rec_conv2d_bn(ts2_up9_concat,32, 3, 3)
   ts2_up9_f = tf.keras.layers.add([ts2_up9_recconv, ts2_up9_concat])
   ts2_conv9 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up9_f)

   ts2_up10_1 = tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts2_conv9)
   ts2_up10_concat = tf.keras.layers.concatenate([ts2_up10_1, conv0_f], axis=channel_axis)
   ts2_up10_concat = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up10_concat)
   ts2_up10_recconv = Rec_conv2d_bn(ts2_up10_concat,16, 3, 3)
   ts2_up10_f = tf.keras.layers.add([ts2_up10_recconv, ts2_up10_concat])
   ts2_conv10 = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up10_f)
   
   # Decoding from fourth laten space. 
   ts1_up6_1 = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv5_f)
   ts1_up6_concat = tf.keras.layers.concatenate([ts1_up6_1, conv4_f], axis=channel_axis)
   ts1_up6_concat = tf.keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up6_concat)
   ts1_up6_recconv = Rec_conv2d_bn(ts1_up6_concat,256, 3, 3)
   ts1_up6_f = tf.keras.layers.add([ts1_up6_recconv, ts1_up6_concat])
   ts1_conv6 = tf.keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up6_f)
   
   ts1_up7_1 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts1_conv6)
   ts1_up7_concat = tf.keras.layers.concatenate([ts1_up7_1, conv3_f], axis=channel_axis)
   ts1_up7_concat = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up7_concat)
   ts1_up7_recconv = Rec_conv2d_bn(ts1_up7_concat,128, 3, 3)
   ts1_up7_f = tf.keras.layers.add([ts1_up7_recconv, ts1_up7_concat])
   ts1_conv7 = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up7_f)

   ts1_up8_1 =tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts1_conv7)
   ts1_up8_concat = tf.keras.layers.concatenate([ts1_up8_1, conv2_f], axis=channel_axis)
   ts1_up8_concat = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up8_concat)
   ts1_up8_recconv = Rec_conv2d_bn(ts1_up8_concat,64, 3, 3)
   ts1_up8_f = tf.keras.layers.add([ts1_up8_recconv, ts1_up8_concat])
   ts1_conv8 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up8_f)
 
   ts1_up9_1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts1_conv8)
   ts1_up9_concat = tf.keras.layers.concatenate([ts1_up9_1, conv1_f], axis=channel_axis)
   ts1_up9_concat = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up9_concat)
   ts1_up9_recconv = Rec_conv2d_bn(ts1_up9_concat,32, 3, 3)
   ts1_up9_f = tf.keras.layers.add([ts1_up9_recconv, ts1_up9_concat])
   ts1_conv9 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up9_f)

   ts1_up10_1 = tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts1_conv9)
   ts1_up10_concat = tf.keras.layers.concatenate([ts1_up10_1, conv0_f], axis=channel_axis)
   ts1_up10_concat = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up10_concat)
   ts1_up10_recconv = Rec_conv2d_bn(ts1_up10_concat,16, 3, 3)
   ts1_up10_f = tf.keras.layers.add([ts1_up10_recconv, ts1_up10_concat])
   ts1_conv10 = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up10_f)
   
   
   
   # conncate laten space first and second
   
   ts12_up10_concat = tf.keras.layers.concatenate([ts1_conv10, ts2_conv10], axis=channel_axis)
   
   #ts123_up10_concat = tf.keras.layers.concatenate([ts12_up10_concat, ts3_conv10], axis=channel_axis)
   
   #ts1234_up10_concat = tf.keras.layers.concatenate([ts123_up10_concat, ts4_conv10], axis=channel_axis)

   model_features = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts12_up10_concat)
   
   if num_classes <= 2:
        model_outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid',data_format='channels_last')(model_features)
   else:
        model_outputs = tf.keras.layers.Conv2D(num_classes,(1,1), activation='softmax')(model_features)
       

   #model = tf.keras.Model(inputs=[inputs], outputs=[conv10])
   model = tf.keras.Model(inputs=inputs, outputs=model_outputs)


   #model.compile(optimizer=Adam(2e-4),loss=dice_coef_loss,metrics = [dice_coef, 'acc', 'mse'])
    
   #model.compile(optimizer=tf.train.AdamOptimizer(3e-4), loss='binary_crossentropy',metrics = ['acc', 'mse'])
    
   return model


def build_DeltaNetB_final(input_shape,num_classes):
      

   channel_axis = 3
   inputs =  tf.keras.Input(input_shape) 

   # zero block
   x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same',data_format='channels_last')(inputs)    
   rcnn_bn0 = Rec_conv2d_bn(x, 16, 3, 3)    
   conv0_f = tf.keras.layers.add([x, rcnn_bn0])

    # downsampling zero block..
   pool0 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv0_f)
   conv_pool0 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool0)

    # first block
   #x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(inputs)    
   rcnn_bn1 = Rec_conv2d_bn(conv_pool0, 32, 3, 3)    
   conv1_f = tf.keras.layers.add([conv_pool0, rcnn_bn1])

    # downsampling first block..
   pool1 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv1_f)
   conv_pool1 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool1)
    
    # second RRCNN layer...
   rcnn_bn2 = Rec_conv2d_bn(conv_pool1, 64, 3, 3)    
   conv2_f = tf.keras.layers.add([conv_pool1, rcnn_bn2])
    
    # downsampling first block..
   pool2 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv2_f)
   conv_pool2 = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool2)
    
    # third RRCNN layer...
   rcnn_bn3 = Rec_conv2d_bn(conv_pool2, 128, 3, 3)    
   conv3_f = tf.keras.layers.add([conv_pool2, rcnn_bn3])
    
    # downsampling first block..
   pool3 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv3_f)
   conv_pool3 = tf.keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool3)
    
    # fourth RRCNN layer...
   rcnn_bn4 = Rec_conv2d_bn(conv_pool3, 256, 3, 3)    
   conv4_f = tf.keras.layers.add([conv_pool3, rcnn_bn4])
   
   # fifth block .....
   pool4 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv4_f)
   conv_pool4 = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool4)
    
    # fourth RRCNN layer...
   rcnn_bn5 = Rec_conv2d_bn(conv_pool4, 512, 3, 3)    
   conv5_f = tf.keras.layers.add([conv_pool4, rcnn_bn5])
     
    # Decoding from first laten space.
    
   # Decoding from seocond latent space....
   '''
   # Decoding from fourth laten space.
   ts4_up9_1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv2_f)
   ts4_up9_concat = tf.keras.layers.concatenate([ts4_up9_1, conv1_f], axis=channel_axis)
   ts4_up9_concat = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts4_up9_concat)
   ts4_up9_recconv = Rec_conv2d_bn(ts4_up9_concat,32, 3, 3)
   ts4_up9_f = tf.keras.layers.add([ts4_up9_recconv, ts4_up9_concat])
   ts4_conv9 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts4_up9_f)

   ts4_up10_1 = tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts4_conv9)
   ts4_up10_concat = tf.keras.layers.concatenate([ts4_up10_1, conv0_f], axis=channel_axis)
   ts4_up10_concat = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts4_up10_concat)
   ts4_up10_recconv = Rec_conv2d_bn(ts4_up10_concat,16, 3, 3)
   ts4_up10_f = tf.keras.layers.add([ts4_up10_recconv, ts4_up10_concat])
   ts4_conv10 = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts4_up10_f)
   
   

    
    # Decoding from third latent space.... 
    
   # Decoding from third laten space.
   ts3_up8_1 =tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv3_f)
   ts3_up8_concat = tf.keras.layers.concatenate([ts3_up8_1, conv2_f], axis=channel_axis)
   ts3_up8_concat = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts3_up8_concat)
   ts3_up8_recconv = Rec_conv2d_bn(ts3_up8_concat,64, 3, 3)
   ts3_up8_f = tf.keras.layers.add([ts3_up8_recconv, ts3_up8_concat])
   ts3_conv8 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts3_up8_f)
 
   ts3_up9_1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts3_conv8)
   ts3_up9_concat = tf.keras.layers.concatenate([ts3_up9_1, conv1_f], axis=channel_axis)
   ts3_up9_concat = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts3_up9_concat)
   ts3_up9_recconv = Rec_conv2d_bn(ts3_up9_concat,32, 3, 3)
   ts3_up9_f = tf.keras.layers.add([ts3_up9_recconv, ts3_up9_concat])
   ts3_conv9 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts3_up9_f)

   ts3_up10_1 = tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts3_conv9)
   ts3_up10_concat = tf.keras.layers.concatenate([ts3_up10_1, conv0_f], axis=channel_axis)
   ts3_up10_concat = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts3_up10_concat)
   ts3_up10_recconv = Rec_conv2d_bn(ts3_up10_concat,16, 3, 3)
   ts3_up10_f = tf.keras.layers.add([ts3_up10_recconv, ts3_up10_concat])
   ts3_conv10 = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts3_up10_f)
   
   '''
   # Decoding from fourth laten space.
   
   ts2_up7_1 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv4_f)
   ts2_up7_concat = tf.keras.layers.concatenate([ts2_up7_1, conv3_f], axis=channel_axis)
   ts2_up7_concat = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up7_concat)
   ts2_up7_recconv = Rec_conv2d_bn(ts2_up7_concat,128, 3, 3)
   ts2_up7_f = tf.keras.layers.add([ts2_up7_recconv, ts2_up7_concat])
   ts2_conv7 = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up7_f)

   ts2_up8_1 =tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts2_conv7)
   ts2_up8_concat = tf.keras.layers.concatenate([ts2_up8_1, conv2_f], axis=channel_axis)
   ts2_up8_concat = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up8_concat)
   ts2_up8_recconv = Rec_conv2d_bn(ts2_up8_concat,64, 3, 3)
   ts2_up8_f = tf.keras.layers.add([ts2_up8_recconv, ts2_up8_concat])
   ts2_conv8 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up8_f)
 
   ts2_up9_1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts2_conv8)
   ts2_up9_concat = tf.keras.layers.concatenate([ts2_up9_1, conv1_f], axis=channel_axis)
   ts2_up9_concat = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up9_concat)
   ts2_up9_recconv = Rec_conv2d_bn(ts2_up9_concat,32, 3, 3)
   ts2_up9_f = tf.keras.layers.add([ts2_up9_recconv, ts2_up9_concat])
   ts2_conv9 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up9_f)

   ts2_up10_1 = tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts2_conv9)
   ts2_up10_concat = tf.keras.layers.concatenate([ts2_up10_1, conv0_f], axis=channel_axis)
   ts2_up10_concat = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up10_concat)
   ts2_up10_recconv = Rec_conv2d_bn(ts2_up10_concat,16, 3, 3)
   ts2_up10_f = tf.keras.layers.add([ts2_up10_recconv, ts2_up10_concat])
   ts2_conv10 = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up10_f)
   
   # Decoding from fifth laten space. 
   
   ts1_up6_1 = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv5_f)
   ts1_up6_concat = tf.keras.layers.concatenate([ts1_up6_1, conv4_f], axis=channel_axis)
   ts1_up6_concat = tf.keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up6_concat)
   ts1_up6_recconv = Rec_conv2d_bn(ts1_up6_concat,256, 3, 3)
   ts1_up6_f = tf.keras.layers.add([ts1_up6_recconv, ts1_up6_concat])
   ts1_conv6 = tf.keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up6_f)
   
   ts1_up7_1 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts1_conv6)
   ts1_up7_concat = tf.keras.layers.concatenate([ts1_up7_1, conv3_f], axis=channel_axis)
   ts1_up7_concat = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up7_concat)
   ts1_up7_recconv = Rec_conv2d_bn(ts1_up7_concat,128, 3, 3)
   ts1_up7_f = tf.keras.layers.add([ts1_up7_recconv, ts1_up7_concat])
   ts1_conv7 = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up7_f)
   
   ts12_up7_f = tf.keras.layers.add([ts1_conv7, ts2_conv7])

   ts1_up8_1 =tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts12_up7_f)
   ts1_up8_concat = tf.keras.layers.concatenate([ts1_up8_1, conv2_f], axis=channel_axis)
   ts1_up8_concat = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up8_concat)
   ts1_up8_recconv = Rec_conv2d_bn(ts1_up8_concat,64, 3, 3)
   ts1_up8_f = tf.keras.layers.add([ts1_up8_recconv, ts1_up8_concat])
   ts1_conv8 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up8_f)
 
   ts12_up8_f = tf.keras.layers.add([ts1_conv8, ts2_conv8])
       
   ts1_up9_1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts12_up8_f)
   ts1_up9_concat = tf.keras.layers.concatenate([ts1_up9_1, conv1_f], axis=channel_axis)
   ts1_up9_concat = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up9_concat)
   ts1_up9_recconv = Rec_conv2d_bn(ts1_up9_concat,32, 3, 3)
   ts1_up9_f = tf.keras.layers.add([ts1_up9_recconv, ts1_up9_concat])
   ts1_conv9 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up9_f)

   ts12_up9_f = tf.keras.layers.add([ts1_conv9, ts2_conv9])

   ts1_up10_1 = tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts12_up9_f)
   ts1_up10_concat = tf.keras.layers.concatenate([ts1_up10_1, conv0_f], axis=channel_axis)
   ts1_up10_concat = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up10_concat)
   ts1_up10_recconv = Rec_conv2d_bn(ts1_up10_concat,16, 3, 3)
   ts1_up10_f = tf.keras.layers.add([ts1_up10_recconv, ts1_up10_concat])
   ts1_conv10 = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up10_f)
   
   ts12_up10_f = tf.keras.layers.add([ts1_conv10, ts2_conv10])
   
   # conncate laten space first and second
   
   #ts12_up10_concat = tf.keras.layers.concatenate([ts1_conv10, ts2_conv10], axis=channel_axis)
   
   #ts123_up10_concat = tf.keras.layers.concatenate([ts12_up10_concat, ts3_conv10], axis=channel_axis)
   
   #ts1234_up10_concat = tf.keras.layers.concatenate([ts123_up10_concat, ts4_conv10], axis=channel_axis)

   model_features = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts12_up10_f)
   
   if num_classes <= 2:
        model_outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid',data_format='channels_last')(model_features)
   else:
        model_outputs = tf.keras.layers.Conv2D(num_classes,(1,1), activation='softmax')(model_features)
       

   #model = tf.keras.Model(inputs=[inputs], outputs=[conv10])
   model = tf.keras.Model(inputs=inputs, outputs=model_outputs)


   #model.compile(optimizer=Adam(2e-4),loss=dice_coef_loss,metrics = [dice_coef, 'acc', 'mse'])
    
   #model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss='binary_crossentropy',metrics = ['acc', 'mse'])
    
   return model

def build_DeltaNetAB_final(input_shape,num_classes):
      

   channel_axis = 3
   inputs =  tf.keras.Input(input_shape) 

   # zero block
   x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same',data_format='channels_last')(inputs)    
   rcnn_bn0 = Rec_conv2d_bn(x, 16, 3, 3)    
   conv0_f = tf.keras.layers.add([x, rcnn_bn0])

    # downsampling zero block..
   pool0 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv0_f)
   conv_pool0 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool0)

    # first block
   #x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(inputs)    
   rcnn_bn1 = Rec_conv2d_bn(conv_pool0, 32, 3, 3)    
   conv1_f = tf.keras.layers.add([conv_pool0, rcnn_bn1])

    # downsampling first block..
   pool1 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv1_f)
   conv_pool1 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool1)
    
    # second RRCNN layer...
   rcnn_bn2 = Rec_conv2d_bn(conv_pool1, 64, 3, 3)    
   conv2_f = tf.keras.layers.add([conv_pool1, rcnn_bn2])
    
    # downsampling first block..
   pool2 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv2_f)
   conv_pool2 = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool2)
    
    # third RRCNN layer...
   rcnn_bn3 = Rec_conv2d_bn(conv_pool2, 128, 3, 3)    
   conv3_f = tf.keras.layers.add([conv_pool2, rcnn_bn3])
    
    # downsampling first block..
   pool3 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv3_f)
   conv_pool3 = tf.keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool3)
    
    # fourth RRCNN layer...
   rcnn_bn4 = Rec_conv2d_bn(conv_pool3, 256, 3, 3)    
   conv4_f = tf.keras.layers.add([conv_pool3, rcnn_bn4])
   
   # fifth block .....
   pool4 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv4_f)
   conv_pool4 = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool4)
    
    # fourth RRCNN layer...
   rcnn_bn5 = Rec_conv2d_bn(conv_pool4, 512, 3, 3)    
   conv5_f = tf.keras.layers.add([conv_pool4, rcnn_bn5])
     
    # Decoding from first laten space.
    
   # Decoding from seocond latent space....
   '''
   # Decoding from fourth laten space.
   ts4_up9_1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv2_f)
   ts4_up9_concat = tf.keras.layers.concatenate([ts4_up9_1, conv1_f], axis=channel_axis)
   ts4_up9_concat = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts4_up9_concat)
   ts4_up9_recconv = Rec_conv2d_bn(ts4_up9_concat,32, 3, 3)
   ts4_up9_f = tf.keras.layers.add([ts4_up9_recconv, ts4_up9_concat])
   ts4_conv9 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts4_up9_f)

   ts4_up10_1 = tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts4_conv9)
   ts4_up10_concat = tf.keras.layers.concatenate([ts4_up10_1, conv0_f], axis=channel_axis)
   ts4_up10_concat = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts4_up10_concat)
   ts4_up10_recconv = Rec_conv2d_bn(ts4_up10_concat,16, 3, 3)
   ts4_up10_f = tf.keras.layers.add([ts4_up10_recconv, ts4_up10_concat])
   ts4_conv10 = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts4_up10_f)
       
    # Decoding from third latent space.... 
    
   # Decoding from third laten space.
   ts3_up8_1 =tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv3_f)
   ts3_up8_concat = tf.keras.layers.concatenate([ts3_up8_1, conv2_f], axis=channel_axis)
   ts3_up8_concat = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts3_up8_concat)
   ts3_up8_recconv = Rec_conv2d_bn(ts3_up8_concat,64, 3, 3)
   ts3_up8_f = tf.keras.layers.add([ts3_up8_recconv, ts3_up8_concat])
   ts3_conv8 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts3_up8_f)
 
   ts3_up9_1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts3_conv8)
   ts3_up9_concat = tf.keras.layers.concatenate([ts3_up9_1, conv1_f], axis=channel_axis)
   ts3_up9_concat = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts3_up9_concat)
   ts3_up9_recconv = Rec_conv2d_bn(ts3_up9_concat,32, 3, 3)
   ts3_up9_f = tf.keras.layers.add([ts3_up9_recconv, ts3_up9_concat])
   ts3_conv9 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts3_up9_f)

   ts3_up10_1 = tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts3_conv9)
   ts3_up10_concat = tf.keras.layers.concatenate([ts3_up10_1, conv0_f], axis=channel_axis)
   ts3_up10_concat = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts3_up10_concat)
   ts3_up10_recconv = Rec_conv2d_bn(ts3_up10_concat,16, 3, 3)
   ts3_up10_f = tf.keras.layers.add([ts3_up10_recconv, ts3_up10_concat])
   ts3_conv10 = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts3_up10_f)
   
   '''
   # Decoding from fourth laten space.
   
   ts2_up7_1 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv4_f)
   ts2_up7_concat = tf.keras.layers.concatenate([ts2_up7_1, conv3_f], axis=channel_axis)
   ts2_up7_concat = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up7_concat)
   ts2_up7_recconv = Rec_conv2d_bn(ts2_up7_concat,128, 3, 3)
   ts2_up7_f = tf.keras.layers.add([ts2_up7_recconv, ts2_up7_concat])
   ts2_conv7 = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up7_f)

   ts2_up8_1 =tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts2_conv7)
   ts2_up8_concat = tf.keras.layers.concatenate([ts2_up8_1, conv2_f], axis=channel_axis)
   ts2_up8_concat = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up8_concat)
   ts2_up8_recconv = Rec_conv2d_bn(ts2_up8_concat,64, 3, 3)
   ts2_up8_f = tf.keras.layers.add([ts2_up8_recconv, ts2_up8_concat])
   ts2_conv8 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up8_f)
 
   ts2_up9_1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts2_conv8)
   ts2_up9_concat = tf.keras.layers.concatenate([ts2_up9_1, conv1_f], axis=channel_axis)
   ts2_up9_concat = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up9_concat)
   ts2_up9_recconv = Rec_conv2d_bn(ts2_up9_concat,32, 3, 3)
   ts2_up9_f = tf.keras.layers.add([ts2_up9_recconv, ts2_up9_concat])
   ts2_conv9 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up9_f)

   ts2_up10_1 = tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts2_conv9)
   ts2_up10_concat = tf.keras.layers.concatenate([ts2_up10_1, conv0_f], axis=channel_axis)
   ts2_up10_concat = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up10_concat)
   ts2_up10_recconv = Rec_conv2d_bn(ts2_up10_concat,16, 3, 3)
   ts2_up10_f = tf.keras.layers.add([ts2_up10_recconv, ts2_up10_concat])
   ts2_conv10 = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up10_f)
   
   # Decoding from fifth laten space. 
   
   ts1_up6_1 = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv5_f)
   ts1_up6_concat = tf.keras.layers.concatenate([ts1_up6_1, conv4_f], axis=channel_axis)
   ts1_up6_concat = tf.keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up6_concat)
   ts1_up6_recconv = Rec_conv2d_bn(ts1_up6_concat,256, 3, 3)
   ts1_up6_f = tf.keras.layers.add([ts1_up6_recconv, ts1_up6_concat])
   ts1_conv6 = tf.keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up6_f)
   
   ts1_up7_1 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts1_conv6)
   ts1_up7_concat = tf.keras.layers.concatenate([ts1_up7_1, conv3_f], axis=channel_axis)
   ts1_up7_concat = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up7_concat)
   ts1_up7_recconv = Rec_conv2d_bn(ts1_up7_concat,128, 3, 3)
   ts1_up7_f = tf.keras.layers.add([ts1_up7_recconv, ts1_up7_concat])
   ts1_conv7 = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up7_f)
   
   ts12_up7_f = tf.keras.layers.add([ts1_conv7, ts2_conv7])

   ts1_up8_1 =tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts12_up7_f)
   ts1_up8_concat = tf.keras.layers.concatenate([ts1_up8_1, conv2_f], axis=channel_axis)
   ts1_up8_concat = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up8_concat)
   ts1_up8_recconv = Rec_conv2d_bn(ts1_up8_concat,64, 3, 3)
   ts1_up8_f = tf.keras.layers.add([ts1_up8_recconv, ts1_up8_concat])
   ts1_conv8 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up8_f)
 
   ts12_up8_f = tf.keras.layers.add([ts1_conv8, ts2_conv8])
       
   ts1_up9_1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts12_up8_f)
   ts1_up9_concat = tf.keras.layers.concatenate([ts1_up9_1, conv1_f], axis=channel_axis)
   ts1_up9_concat = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up9_concat)
   ts1_up9_recconv = Rec_conv2d_bn(ts1_up9_concat,32, 3, 3)
   ts1_up9_f = tf.keras.layers.add([ts1_up9_recconv, ts1_up9_concat])
   ts1_conv9 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up9_f)

   ts12_up9_f = tf.keras.layers.add([ts1_conv9, ts2_conv9])

   ts1_up10_1 = tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts12_up9_f)
   ts1_up10_concat = tf.keras.layers.concatenate([ts1_up10_1, conv0_f], axis=channel_axis)
   ts1_up10_concat = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up10_concat)
   ts1_up10_recconv = Rec_conv2d_bn(ts1_up10_concat,16, 3, 3)
   ts1_up10_f = tf.keras.layers.add([ts1_up10_recconv, ts1_up10_concat])
   ts1_conv10 = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up10_f)
   
   ts12_up10_f = tf.keras.layers.add([ts1_conv10, ts2_conv10])
   
   # conncate laten space first and second

   ts12_up10_concat = tf.keras.layers.concatenate([ts12_up10_f, ts2_conv10], axis=channel_axis)   
   #ts123_up10_concat = tf.keras.layers.concatenate([ts12_up10_concat, ts3_conv10], axis=channel_axis)
   
   #ts1234_up10_concat = tf.keras.layers.concatenate([ts123_up10_concat, ts4_conv10], axis=channel_axis)

   model_features = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts12_up10_concat)
   
   if num_classes <= 2:
        model_outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid',data_format='channels_last')(model_features)
   else:
        model_outputs = tf.keras.layers.Conv2D(num_classes,(1,1), activation='softmax')(model_features)
       

   #model = tf.keras.Model(inputs=[inputs], outputs=[conv10])
   model = tf.keras.Model(inputs=inputs, outputs=model_outputs)


   #model.compile(optimizer=Adam(2e-4),loss=dice_coef_loss,metrics = [dice_coef, 'acc', 'mse'])
    
   model.compile(optimizer=tf.train.AdamOptimizer(3e-4), loss='binary_crossentropy',metrics = ['acc', 'mse'])
    
   return model


def build_DeltaNetAB_3fs_final(input_shape,num_classes):
      

   channel_axis = 3
   inputs =  tf.keras.Input(input_shape) 

   # zero block
   x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same',data_format='channels_last')(inputs)    
   rcnn_bn0 = Rec_conv2d_bn(x, 16, 3, 3)    
   conv0_f = tf.keras.layers.add([x, rcnn_bn0])

    # downsampling zero block..
   pool0 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv0_f)
   conv_pool0 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool0)

    # first block
   #x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(inputs)    
   rcnn_bn1 = Rec_conv2d_bn(conv_pool0, 32, 3, 3)    
   conv1_f = tf.keras.layers.add([conv_pool0, rcnn_bn1])

    # downsampling first block..
   pool1 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv1_f)
   conv_pool1 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool1)
    
    # second RRCNN layer...
   rcnn_bn2 = Rec_conv2d_bn(conv_pool1, 64, 3, 3)    
   conv2_f = tf.keras.layers.add([conv_pool1, rcnn_bn2])
    
    # downsampling first block..
   pool2 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv2_f)
   conv_pool2 = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool2)
    
    # third RRCNN layer...
   rcnn_bn3 = Rec_conv2d_bn(conv_pool2, 128, 3, 3)    
   conv3_f = tf.keras.layers.add([conv_pool2, rcnn_bn3])
    
    # downsampling first block..
   pool3 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv3_f)
   conv_pool3 = tf.keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool3)
    
    # fourth RRCNN layer...
   rcnn_bn4 = Rec_conv2d_bn(conv_pool3, 256, 3, 3)    
   conv4_f = tf.keras.layers.add([conv_pool3, rcnn_bn4])
   
   # fifth block .....
   pool4 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv4_f)
   conv_pool4 = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool4)
    
    # fourth RRCNN layer...
   rcnn_bn5 = Rec_conv2d_bn(conv_pool4, 512, 3, 3)    
   conv5_f = tf.keras.layers.add([conv_pool4, rcnn_bn5])
     
    # Decoding from first laten space.
    
   # Decoding from seocond latent space....
   '''
   # Decoding from fourth laten space.
   ts4_up9_1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv2_f)
   ts4_up9_concat = tf.keras.layers.concatenate([ts4_up9_1, conv1_f], axis=channel_axis)
   ts4_up9_concat = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts4_up9_concat)
   ts4_up9_recconv = Rec_conv2d_bn(ts4_up9_concat,32, 3, 3)
   ts4_up9_f = tf.keras.layers.add([ts4_up9_recconv, ts4_up9_concat])
   ts4_conv9 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts4_up9_f)

   ts4_up10_1 = tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts4_conv9)
   ts4_up10_concat = tf.keras.layers.concatenate([ts4_up10_1, conv0_f], axis=channel_axis)
   ts4_up10_concat = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts4_up10_concat)
   ts4_up10_recconv = Rec_conv2d_bn(ts4_up10_concat,16, 3, 3)
   ts4_up10_f = tf.keras.layers.add([ts4_up10_recconv, ts4_up10_concat])
   ts4_conv10 = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts4_up10_f)
   '''    
    # Decoding from third latent space.... 
    
    
   # Decoding from third laten space.
   ts3_up8_1 =tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv3_f)
   ts3_up8_concat = tf.keras.layers.concatenate([ts3_up8_1, conv2_f], axis=channel_axis)
   ts3_up8_concat = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts3_up8_concat)
   ts3_up8_recconv = Rec_conv2d_bn(ts3_up8_concat,64, 3, 3)
   ts3_up8_f = tf.keras.layers.add([ts3_up8_recconv, ts3_up8_concat])
   ts3_conv8 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts3_up8_f)
 
   ts3_up9_1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts3_conv8)
   ts3_up9_concat = tf.keras.layers.concatenate([ts3_up9_1, conv1_f], axis=channel_axis)
   ts3_up9_concat = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts3_up9_concat)
   ts3_up9_recconv = Rec_conv2d_bn(ts3_up9_concat,32, 3, 3)
   ts3_up9_f = tf.keras.layers.add([ts3_up9_recconv, ts3_up9_concat])
   ts3_conv9 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts3_up9_f)

   ts3_up10_1 = tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts3_conv9)
   ts3_up10_concat = tf.keras.layers.concatenate([ts3_up10_1, conv0_f], axis=channel_axis)
   ts3_up10_concat = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts3_up10_concat)
   ts3_up10_recconv = Rec_conv2d_bn(ts3_up10_concat,16, 3, 3)
   ts3_up10_f = tf.keras.layers.add([ts3_up10_recconv, ts3_up10_concat])
   ts3_conv10 = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts3_up10_f)
   

   # Decoding from fourth laten space.
   
   ts2_up7_1 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv4_f)
   ts2_up7_concat = tf.keras.layers.concatenate([ts2_up7_1, conv3_f], axis=channel_axis)
   ts2_up7_concat = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up7_concat)
   ts2_up7_recconv = Rec_conv2d_bn(ts2_up7_concat,128, 3, 3)
   ts2_up7_f = tf.keras.layers.add([ts2_up7_recconv, ts2_up7_concat])
   ts2_conv7 = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up7_f)

   ts2_up8_1 =tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts2_conv7)
   ts2_up8_concat = tf.keras.layers.concatenate([ts2_up8_1, conv2_f], axis=channel_axis)
   ts2_up8_concat = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up8_concat)
   ts2_up8_recconv = Rec_conv2d_bn(ts2_up8_concat,64, 3, 3)
   ts2_up8_f = tf.keras.layers.add([ts2_up8_recconv, ts2_up8_concat])
   ts2_conv8 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up8_f)
   
   ts23_up8_f = tf.keras.layers.add([ts2_conv8, ts3_conv8])
 
   ts2_up9_1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts23_up8_f)
   ts2_up9_concat = tf.keras.layers.concatenate([ts2_up9_1, conv1_f], axis=channel_axis)
   ts2_up9_concat = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up9_concat)
   ts2_up9_recconv = Rec_conv2d_bn(ts2_up9_concat,32, 3, 3)
   ts2_up9_f = tf.keras.layers.add([ts2_up9_recconv, ts2_up9_concat])
   ts2_conv9 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up9_f)

   ts23_up9_f = tf.keras.layers.add([ts2_conv9, ts3_conv9])
   
   ts2_up10_1 = tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts23_up9_f)
   ts2_up10_concat = tf.keras.layers.concatenate([ts2_up10_1, conv0_f], axis=channel_axis)
   ts2_up10_concat = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up10_concat)
   ts2_up10_recconv = Rec_conv2d_bn(ts2_up10_concat,16, 3, 3)
   ts2_up10_f = tf.keras.layers.add([ts2_up10_recconv, ts2_up10_concat])
   ts2_conv10 = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up10_f)
   
   #ts23_conv10_f = tf.keras.layers.add([ts2_conv10, ts3_conv10])
   # Decoding from fifth laten space. 
   
   ts1_up6_1 = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv5_f)
   ts1_up6_concat = tf.keras.layers.concatenate([ts1_up6_1, conv4_f], axis=channel_axis)
   ts1_up6_concat = tf.keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up6_concat)
   ts1_up6_recconv = Rec_conv2d_bn(ts1_up6_concat,256, 3, 3)
   ts1_up6_f = tf.keras.layers.add([ts1_up6_recconv, ts1_up6_concat])
   ts1_conv6 = tf.keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up6_f)
   
   ts1_up7_1 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts1_conv6)
   ts1_up7_concat = tf.keras.layers.concatenate([ts1_up7_1, conv3_f], axis=channel_axis)
   ts1_up7_concat = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up7_concat)
   ts1_up7_recconv = Rec_conv2d_bn(ts1_up7_concat,128, 3, 3)
   ts1_up7_f = tf.keras.layers.add([ts1_up7_recconv, ts1_up7_concat])
   ts1_conv7 = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up7_f)
   
   ts12_up7_f = tf.keras.layers.add([ts1_conv7, ts2_conv7])

   ts1_up8_1 =tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts12_up7_f)
   ts1_up8_concat = tf.keras.layers.concatenate([ts1_up8_1, conv2_f], axis=channel_axis)
   ts1_up8_concat = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up8_concat)
   ts1_up8_recconv = Rec_conv2d_bn(ts1_up8_concat,64, 3, 3)
   ts1_up8_f = tf.keras.layers.add([ts1_up8_recconv, ts1_up8_concat])
   ts1_conv8 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up8_f)
 
   ts12_up8_f = tf.keras.layers.add([ts1_conv8, ts2_conv8])
       
   ts1_up9_1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts12_up8_f)
   ts1_up9_concat = tf.keras.layers.concatenate([ts1_up9_1, conv1_f], axis=channel_axis)
   ts1_up9_concat = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up9_concat)
   ts1_up9_recconv = Rec_conv2d_bn(ts1_up9_concat,32, 3, 3)
   ts1_up9_f = tf.keras.layers.add([ts1_up9_recconv, ts1_up9_concat])
   ts1_conv9 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up9_f)

   ts12_up9_f = tf.keras.layers.add([ts1_conv9, ts2_conv9])

   ts1_up10_1 = tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts12_up9_f)
   ts1_up10_concat = tf.keras.layers.concatenate([ts1_up10_1, conv0_f], axis=channel_axis)
   ts1_up10_concat = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up10_concat)
   ts1_up10_recconv = Rec_conv2d_bn(ts1_up10_concat,16, 3, 3)
   ts1_up10_f = tf.keras.layers.add([ts1_up10_recconv, ts1_up10_concat])
   ts1_conv10 = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up10_f)
   
   ts12_up10_f = tf.keras.layers.add([ts1_conv10, ts2_conv10])
   
   ts12_up10_f_con = tf.keras.layers.concatenate([ts12_up10_f,ts1_conv10])
   
   # conncate laten space first and second

   ts12_up10_concat = tf.keras.layers.concatenate([ts12_up10_f_con, ts2_conv10], axis=channel_axis)   
   ts123_up10_concat = tf.keras.layers.concatenate([ts12_up10_concat, ts3_conv10], axis=channel_axis)
   
   #ts1234_up10_concat = tf.keras.layers.concatenate([ts123_up10_concat, ts4_conv10], axis=channel_axis)

   model_features = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts123_up10_concat)
   
   if num_classes <= 2:
        model_outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid',data_format='channels_last')(model_features)
   else:
        model_outputs = tf.keras.layers.Conv2D(num_classes,(1,1), activation='softmax')(model_features)
       

   #model = tf.keras.Model(inputs=[inputs], outputs=[conv10])
   model = tf.keras.Model(inputs=inputs, outputs=model_outputs)


   #model.compile(optimizer=Adam(2e-4),loss=dice_coef_loss,metrics = [dice_coef, 'acc', 'mse'])
    
   #model.compile(optimizer=tf.train.AdamOptimizer(3e-4), loss='binary_crossentropy',metrics = ['acc', 'mse'])
    
   return model

def build_DeltaNetAB_4fs_final(input_shape,num_classes):
      

   channel_axis = 3
   inputs =  tf.keras.Input(input_shape) 

   # zero block
   x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same',data_format='channels_last')(inputs)    
   rcnn_bn0 = Rec_conv2d_bn(x, 16, 3, 3)    
   conv0_f = tf.keras.layers.add([x, rcnn_bn0])

    # downsampling zero block..
   pool0 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv0_f)
   conv_pool0 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool0)

    # first block
   #x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(inputs)    
   rcnn_bn1 = Rec_conv2d_bn(conv_pool0, 32, 3, 3)    
   conv1_f = tf.keras.layers.add([conv_pool0, rcnn_bn1])

    # downsampling first block..
   pool1 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv1_f)
   conv_pool1 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool1)
    
    # second RRCNN layer...
   rcnn_bn2 = Rec_conv2d_bn(conv_pool1, 64, 3, 3)    
   conv2_f = tf.keras.layers.add([conv_pool1, rcnn_bn2])
    
    # downsampling first block..
   pool2 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv2_f)
   conv_pool2 = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool2)
    
    # third RRCNN layer...
   rcnn_bn3 = Rec_conv2d_bn(conv_pool2, 128, 3, 3)    
   conv3_f = tf.keras.layers.add([conv_pool2, rcnn_bn3])
    
    # downsampling first block..
   pool3 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv3_f)
   conv_pool3 = tf.keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool3)
    
    # fourth RRCNN layer...
   rcnn_bn4 = Rec_conv2d_bn(conv_pool3, 256, 3, 3)    
   conv4_f = tf.keras.layers.add([conv_pool3, rcnn_bn4])
   
   # fifth block .....
   pool4 = tf.keras.layers.MaxPooling2D((2, 2),data_format='channels_last')(conv4_f)
   conv_pool4 = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool4)
    
    # fourth RRCNN layer...
   rcnn_bn5 = Rec_conv2d_bn(conv_pool4, 512, 3, 3)    
   conv5_f = tf.keras.layers.add([conv_pool4, rcnn_bn5])
     
    # Decoding from first laten space.
    
   # Decoding from seocond latent space....
   
   # Decoding from fourth laten space.
   ts4_up9_1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv2_f)
   ts4_up9_concat = tf.keras.layers.concatenate([ts4_up9_1, conv1_f], axis=channel_axis)
   ts4_up9_concat = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts4_up9_concat)
   ts4_up9_recconv = Rec_conv2d_bn(ts4_up9_concat,32, 3, 3)
   ts4_up9_f = tf.keras.layers.add([ts4_up9_recconv, ts4_up9_concat])
   ts4_conv9 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts4_up9_f)

   ts4_up10_1 = tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts4_conv9)
   ts4_up10_concat = tf.keras.layers.concatenate([ts4_up10_1, conv0_f], axis=channel_axis)
   ts4_up10_concat = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts4_up10_concat)
   ts4_up10_recconv = Rec_conv2d_bn(ts4_up10_concat,16, 3, 3)
   ts4_up10_f = tf.keras.layers.add([ts4_up10_recconv, ts4_up10_concat])
   ts4_conv10 = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts4_up10_f)
       
    # Decoding from third latent space.... 
    
    
   # Decoding from third laten space.
   ts3_up8_1 =tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv3_f)
   ts3_up8_concat = tf.keras.layers.concatenate([ts3_up8_1, conv2_f], axis=channel_axis)
   ts3_up8_concat = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts3_up8_concat)
   ts3_up8_recconv = Rec_conv2d_bn(ts3_up8_concat,64, 3, 3)
   ts3_up8_f = tf.keras.layers.add([ts3_up8_recconv, ts3_up8_concat])
   ts3_conv8 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts3_up8_f)
 
   ts3_up9_1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts3_conv8)
   ts3_up9_concat = tf.keras.layers.concatenate([ts3_up9_1, conv1_f], axis=channel_axis)
   ts3_up9_concat = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts3_up9_concat)
   ts3_up9_recconv = Rec_conv2d_bn(ts3_up9_concat,32, 3, 3)
   ts3_up9_f = tf.keras.layers.add([ts3_up9_recconv, ts3_up9_concat])
   ts3_conv9 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts3_up9_f)
   
   #ts34_up8_f = tf.keras.layers.add([ts3_conv9, ts4_conv9])

   ts3_up10_1 = tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts3_conv9)
   ts3_up10_concat = tf.keras.layers.concatenate([ts3_up10_1, conv0_f], axis=channel_axis)
   ts3_up10_concat = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts3_up10_concat)
   ts3_up10_recconv = Rec_conv2d_bn(ts3_up10_concat,16, 3, 3)
   ts3_up10_f = tf.keras.layers.add([ts3_up10_recconv, ts3_up10_concat])
   ts3_conv10 = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts3_up10_f)
   

   #ts34_conv10_f = tf.keras.layers.add([ts4_conv10, ts4_conv10])
   
   # Decoding from fourth laten space.
   
   ts2_up7_1 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv4_f)
   ts2_up7_concat = tf.keras.layers.concatenate([ts2_up7_1, conv3_f], axis=channel_axis)
   ts2_up7_concat = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up7_concat)
   ts2_up7_recconv = Rec_conv2d_bn(ts2_up7_concat,128, 3, 3)
   ts2_up7_f = tf.keras.layers.add([ts2_up7_recconv, ts2_up7_concat])
   ts2_conv7 = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up7_f)

   ts2_up8_1 =tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts2_conv7)
   ts2_up8_concat = tf.keras.layers.concatenate([ts2_up8_1, conv2_f], axis=channel_axis)
   ts2_up8_concat = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up8_concat)
   ts2_up8_recconv = Rec_conv2d_bn(ts2_up8_concat,64, 3, 3)
   ts2_up8_f = tf.keras.layers.add([ts2_up8_recconv, ts2_up8_concat])
   ts2_conv8 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up8_f)
   
   ts23_up8_f = tf.keras.layers.add([ts2_conv8, ts3_conv8])
 
   ts2_up9_1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts23_up8_f)
   ts2_up9_concat = tf.keras.layers.concatenate([ts2_up9_1, conv1_f], axis=channel_axis)
   ts2_up9_concat = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up9_concat)
   ts2_up9_recconv = Rec_conv2d_bn(ts2_up9_concat,32, 3, 3)
   ts2_up9_f = tf.keras.layers.add([ts2_up9_recconv, ts2_up9_concat])
   ts2_conv9 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up9_f)

   ts23_up9_f = tf.keras.layers.add([ts2_conv9, ts3_conv9])
   
   ts2_up10_1 = tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts23_up9_f)
   ts2_up10_concat = tf.keras.layers.concatenate([ts2_up10_1, conv0_f], axis=channel_axis)
   ts2_up10_concat = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up10_concat)
   ts2_up10_recconv = Rec_conv2d_bn(ts2_up10_concat,16, 3, 3)
   ts2_up10_f = tf.keras.layers.add([ts2_up10_recconv, ts2_up10_concat])
   ts2_conv10 = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts2_up10_f)
   
   #ts23_conv10_f = tf.keras.layers.add([ts2_conv10, ts34_conv10_f])
   
   # Decoding from fifth laten space. 
   
   ts1_up6_1 = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv5_f)
   ts1_up6_concat = tf.keras.layers.concatenate([ts1_up6_1, conv4_f], axis=channel_axis)
   ts1_up6_concat = tf.keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up6_concat)
   ts1_up6_recconv = Rec_conv2d_bn(ts1_up6_concat,256, 3, 3)
   ts1_up6_f = tf.keras.layers.add([ts1_up6_recconv, ts1_up6_concat])
   ts1_conv6 = tf.keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up6_f)
   
   ts1_up7_1 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts1_conv6)
   ts1_up7_concat = tf.keras.layers.concatenate([ts1_up7_1, conv3_f], axis=channel_axis)
   ts1_up7_concat = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up7_concat)
   ts1_up7_recconv = Rec_conv2d_bn(ts1_up7_concat,128, 3, 3)
   ts1_up7_f = tf.keras.layers.add([ts1_up7_recconv, ts1_up7_concat])
   ts1_conv7 = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up7_f)
   
   ts12_up7_f = tf.keras.layers.add([ts1_conv7, ts2_conv7])

   ts1_up8_1 =tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts12_up7_f)
   ts1_up8_concat = tf.keras.layers.concatenate([ts1_up8_1, conv2_f], axis=channel_axis)
   ts1_up8_concat = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up8_concat)
   ts1_up8_recconv = Rec_conv2d_bn(ts1_up8_concat,64, 3, 3)
   ts1_up8_f = tf.keras.layers.add([ts1_up8_recconv, ts1_up8_concat])
   ts1_conv8 = tf.keras.layers.Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up8_f)
 
   ts12_up8_f = tf.keras.layers.add([ts1_conv8, ts2_conv8])
       
   ts1_up9_1 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts12_up8_f)
   ts1_up9_concat = tf.keras.layers.concatenate([ts1_up9_1, conv1_f], axis=channel_axis)
   ts1_up9_concat = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up9_concat)
   ts1_up9_recconv = Rec_conv2d_bn(ts1_up9_concat,32, 3, 3)
   ts1_up9_f = tf.keras.layers.add([ts1_up9_recconv, ts1_up9_concat])
   ts1_conv9 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up9_f)

   ts12_up9_f = tf.keras.layers.add([ts1_conv9, ts2_conv9])

   ts1_up10_1 = tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(ts12_up9_f)
   ts1_up10_concat = tf.keras.layers.concatenate([ts1_up10_1, conv0_f], axis=channel_axis)
   ts1_up10_concat = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up10_concat)
   ts1_up10_recconv = Rec_conv2d_bn(ts1_up10_concat,16, 3, 3)
   ts1_up10_f = tf.keras.layers.add([ts1_up10_recconv, ts1_up10_concat])
   ts1_conv10 = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1_up10_f)
   
   ts12_up10_f = tf.keras.layers.add([ts1_conv10, ts2_conv10])
   
   # conncate laten space first and second

   ts12_up10_concat = tf.keras.layers.concatenate([ts12_up10_f, ts2_conv10], axis=channel_axis)   
   ts123_up10_concat = tf.keras.layers.concatenate([ts12_up10_concat, ts3_conv10], axis=channel_axis)
   
   ts1234_up10_concat = tf.keras.layers.concatenate([ts123_up10_concat, ts4_conv10], axis=channel_axis)

   model_features = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', padding='same',data_format='channels_last')(ts1234_up10_concat)
   
   if num_classes <= 2:
        model_outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid',data_format='channels_last')(model_features)
   else:
        model_outputs = tf.keras.layers.Conv2D(num_classes,(1,1), activation='softmax')(model_features)
       

   #model = tf.keras.Model(inputs=[inputs], outputs=[conv10])
   model = tf.keras.Model(inputs=inputs, outputs=model_outputs)


   #model.compile(optimizer=Adam(2e-4),loss=dice_coef_loss,metrics = [dice_coef, 'acc', 'mse'])
    
   #model.compile(optimizer=tf.train.AdamOptimizer(3e-4), loss='binary_crossentropy',metrics = ['acc', 'mse'])
    
   return model