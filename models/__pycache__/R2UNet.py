# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 22:25:17 2018
# Reference:
Alom, Md Zahangir, et al. "Recurrent residual convolutional neural network based on u-net (r2u-net) for medical image segmentation." arXiv preprint arXiv:1802.06955 (2018).
@author: zahangir
"""
from __future__ import print_function
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate,ZeroPadding2D, Conv2DTranspose,Conv2D,Activation, MaxPooling2D, UpSampling2D, Reshape, core, Dropout,add,merge
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.optimizers import SGD
from keras.regularizers import l2
import matplotlib.pyplot as plt

smooth = 1.

# Metric function
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Loss funtion
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)



def Rec_conv2d_bn(x, nb_filter, nb_row, nb_col,border_mode='same',strides=(1, 1),name=None):
    '''Utility function to apply conv + BN.
    '''
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
        
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
        
    x1 = Conv2D(nb_filter, (nb_row, nb_col),strides=strides,
                      padding=border_mode,
                      name=conv_name,kernel_regularizer=l2(0.0002),data_format='channels_last')(x)
    x1=Activation('relu')(x1)                  
    
    x2 = Conv2D(nb_filter, (nb_row, nb_col),strides=strides,
                      padding=border_mode,
                      name=conv_name,kernel_regularizer=l2(0.0002),data_format='channels_last')(x1)              
    x2=Activation('relu')(x2)                   
    x12 = add([x1, x2])
    
    x3 = Conv2D(nb_filter, (nb_row, nb_col),strides=strides,
                      padding=border_mode,
                      name=conv_name,kernel_regularizer=l2(0.0002),data_format='channels_last')(x12) 
    x3=Activation('relu')(x3)      
    x13 = add([x1, x3])

    x4 = Conv2D(nb_filter, (nb_row, nb_col),strides=strides,
                      padding='same',
                      name=conv_name,kernel_regularizer=l2(0.0002),data_format='channels_last')(x13)
    x4=Activation('relu')(x4) 
   
    x14 = add([x1, x4])
    
    x=Activation('relu')(x14)
    
    #x = BatchNormalization(axis=bn_axis, name=bn_name)(x13)
    return x

#Define the neural network

    

def build_R2UNetE(input_shape,num_classes):
    
      
    inputs = Input(input_shape)   
    channel_axis = 3
    #pdb.set_trace()
    # first block
    x = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(inputs)    
    rcnn_bn1 = Rec_conv2d_bn(x, 32, 3, 3)    
    conv1_f = add([x, rcnn_bn1])

    # downsampling first block..
    pool1 = MaxPooling2D((2, 2),data_format='channels_last')(conv1_f)
    conv_pool1 = Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool1)
    
    # second RRCNN layer...
    rcnn_bn2 = Rec_conv2d_bn(conv_pool1, 64, 3, 3)    
    conv2_f = add([conv_pool1, rcnn_bn2])
    
    # downsampling first block..
    pool2 = MaxPooling2D((2, 2),data_format='channels_last')(conv2_f)
    conv_pool2 = Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool2)
    
    # third RRCNN layer...
    rcnn_bn3 = Rec_conv2d_bn(conv_pool2, 128, 3, 3)    
    conv3_f = add([conv_pool2, rcnn_bn3])
    
    # downsampling first block..
    pool3 = MaxPooling2D((2, 2),data_format='channels_last')(conv3_f)
    conv_pool3 = Conv2D(256, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool3)
    
    # fourth RRCNN layer...
    rcnn_bn4 = Rec_conv2d_bn(conv_pool3, 256, 3, 3)    
    conv4_f = add([conv_pool3, rcnn_bn4])
      
    # Decoder...
   
    #pdb.set_trace()
    
    up7_1 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv4_f)
    up7 = concatenate([up7_1, conv3_f], axis=channel_axis)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv7)
    
    up8_1 =Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv7)
    up8 = concatenate([up8_1, conv2_f], axis=channel_axis)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv8)
    
    up9_1 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv8)
    up9 = concatenate([up9_1, conv1_f], axis=channel_axis)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv9)
    
    conv10 = Conv2D(1, (1, 1), activation='sigmoid',data_format='channels_last')(conv9)
       
#    conv10 = core.Reshape((12,patch_height*patch_width))(conv10)
#    conv10 = core.Permute((2,1))(conv10)
#    conv10 = core.Activation('softmax')(conv10)

    model = Model(inputs=[inputs], outputs=[conv10])
    # Adam(lr=1e-5)


    #model.compile(optimizer=Adam(2e-4),loss=dice_coef_loss,metrics = [dice_coef, 'acc', 'mse'])
    
    model.compile(optimizer=Adam(lr=3e-4), loss='binary_crossentropy',metrics = ['acc', 'mse'])

    
    return model 


def build_R2UNetED(input_shape,num_classes):
    
   """    
    if input_tensor is None:
        final_input = Input(input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            final_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            final_input = input_tensor
    """      
   inputs = Input(input_shape) 
   channel_axis = 3
    
    #pdb.set_trace()
    # first block
   x = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(inputs)    
   rcnn_bn1 = Rec_conv2d_bn(x, 32, 3, 3)    
   conv1_f = add([x, rcnn_bn1])

    # downsampling first block..
   pool1 = MaxPooling2D((2, 2),data_format='channels_last')(conv1_f)
   conv_pool1 = Conv2D(64, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool1)
    
    # second RRCNN layer...
   rcnn_bn2 = Rec_conv2d_bn(conv_pool1, 64, 3, 3)    
   conv2_f = add([conv_pool1, rcnn_bn2])
    
    # downsampling first block..
   pool2 = MaxPooling2D((2, 2),data_format='channels_last')(conv2_f)
   conv_pool2 = Conv2D(128, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool2)
    
    # third RRCNN layer...
   rcnn_bn3 = Rec_conv2d_bn(conv_pool2, 128, 3, 3)    
   conv3_f = add([conv_pool2, rcnn_bn3])
    
    # downsampling first block..
   pool3 = MaxPooling2D((2, 2),data_format='channels_last')(conv3_f)
   conv_pool3 = Conv2D(256, (1, 1), activation='relu', padding='same',data_format='channels_last')(pool3)
    
    # fourth RRCNN layer...
   rcnn_bn4 = Rec_conv2d_bn(conv_pool3, 256, 3, 3)    
   conv4_f = add([conv_pool3, rcnn_bn4])
      
    # Decoder...
   
    #pdb.set_trace()
    
   up7_1 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv4_f)
   up7 = concatenate([up7_1, conv3_f], axis=channel_axis)
   up7 = Rec_conv2d_bn(up7,128, 3, 3)
   conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(up7)
    #conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv7)
    
   up8_1 =Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv7)
   up8 = concatenate([up8_1, conv2_f], axis=channel_axis)
   up8 = Rec_conv2d_bn(up8,128, 3, 3)
   conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(up8)
    #conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv8)
    
   up9_1 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',data_format='channels_last')(conv8)
   up9 = concatenate([up9_1, conv1_f], axis=channel_axis)
   up9 = Rec_conv2d_bn(up9,128, 3, 3)
   conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(up9)
    #conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(conv9)
    
   conv10 = Conv2D(1, (1, 1), activation='sigmoid',data_format='channels_last')(conv9)
       
#    conv10 = core.Reshape((12,patch_height*patch_width))(conv10)
#    conv10 = core.Permute((2,1))(conv10)
#    conv10 = core.Activation('softmax')(conv10)

   model = Model(inputs=[inputs], outputs=[conv10])
    # Adam(lr=1e-5)


    #model.compile(optimizer=Adam(2e-4),loss=dice_coef_loss,metrics = [dice_coef, 'acc', 'mse'])
    
   model.compile(optimizer=Adam(lr=3e-4), loss='binary_crossentropy',metrics = ['acc', 'mse'])

    
   return model 
