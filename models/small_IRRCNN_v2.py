#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Codes for small scale implementation of IRRCNN model by Md Zahangir Alom
# Reference:
- [Improved Inception-Residual Convolutional Neural Network for Object Recognition](https://arxiv.org/abs/1712.09888)
"""
from __future__ import print_function

import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x
    
    
def conv2d_bn(x, nb_filter, nb_row, nb_col,
              border_mode='same',strides=(1, 1),
              name=None):
    '''Utility function to apply conv + BN.
    '''
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    bn_axis = 3

    x = tf.keras.layers.Conv2D(nb_filter, (nb_row, nb_col),strides=strides,
                      padding=border_mode,
                      name=conv_name,kernel_regularizer=tf.keras.regularizers.l2(0.0002),data_format='channels_last')(x)
    x=tf.keras.layers.Activation('relu')(x) 
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)
    return x
    

def Rec_conv2d_bn(x, nb_filter, nb_row, nb_col,
              border_mode='same',strides=(1, 1), 
              name=None):
    '''Utility function to apply conv + BN.
    '''
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
   
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
    
    #x3 = tf.keras.layers.Conv2D(nb_filter, (nb_row, nb_col),strides=strides,
    #                  padding='same',
    #                  name=conv_name,kernel_regularizer=tf.keras.regularizers.l2(0.0002),data_format='channels_last')(x12) 
    #x3=tf.keras.layers.Activation('relu')(x3)      
    #x13 = tf.keras.layers.add([x1, x3])

    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name)(x13)
       
    return x

def build_small_IRRCNN(input_shape,nm_classes):
    
    channel_axis = -1
    img_input = tf.keras.Input(shape=input_shape)
     
    x = conv2d_bn(img_input, 32, 3, 3, border_mode='valid',) # 30x30
    x = conv2d_bn(x, 48, 3, 3, strides=(1, 1), border_mode='valid')
    x = conv2d_bn(x, 64, 3, 3)

    x = conv2d_bn(x, 80, 1, 1, border_mode='valid')  
    x = conv2d_bn(x, 192, 3, 3, border_mode='valid')
    x = conv2d_bn(x, 192, 3, 3, strides=(2, 2), border_mode='valid') # 14x14

    # IRRCNN block # 1
    branch1x1 = Rec_conv2d_bn(x, 32, 1, 1)

    branch5x5 = Rec_conv2d_bn(x, 48, 1, 1)
    branch5x5 = Rec_conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = Rec_conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = Rec_conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = Rec_conv2d_bn(branch3x3dbl, 64, 3, 3)

    branch_pool = tf.keras.layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = Rec_conv2d_bn(branch_pool, 32, 1, 1)
    IR_out1 = tf.keras.layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                  axis=channel_axis,name='mixed0') 
    x = tf.keras.layers.add([x, IR_out1])
    x = tf.keras.layers.Activation('relu')(x)
    
    # Output size: 14x14x192  
    # End of first IRRCNN block.....   
   
    branch3x3 = conv2d_bn(x, 128, 3, 3, strides=(2, 2), border_mode='valid')
    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 64, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 64, 3, 3,
                             strides=(2, 2), border_mode='valid')

    branch_pool = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = tf.keras.layers.concatenate([branch3x3, branch3x3dbl, branch_pool],
              axis=channel_axis,
              name='mixed3')

    # outputs: 14 x 14 x 256
    # for i in range(2):
    branch1x1 = Rec_conv2d_bn(x, 64, 1, 1)

    branch7x7 = Rec_conv2d_bn(x, 160, 1, 1)
    branch7x7 = Rec_conv2d_bn(branch7x7, 160, 1, 7)
    branch7x7 = Rec_conv2d_bn(branch7x7, 64, 7, 1)

    branch7x7dbl = Rec_conv2d_bn(x, 160, 1, 1)
    branch7x7dbl = Rec_conv2d_bn(branch7x7dbl, 160, 7, 1)
    branch7x7dbl = Rec_conv2d_bn(branch7x7dbl, 160, 1, 7)
    branch7x7dbl = Rec_conv2d_bn(branch7x7dbl, 160, 7, 1)
    branch7x7dbl = Rec_conv2d_bn(branch7x7dbl, 64, 1, 7)

    branch_pool = tf.keras.layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = Rec_conv2d_bn(branch_pool, 64, 1, 1)
    IR_out2 = tf.keras.layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                  axis=channel_axis)
    x = tf.keras.layers.add([x, IR_out2])
    x = tf.keras.layers.Activation('relu')(x)
    
    # End of second IRRCNN block.....

    # mixed 8: 7 x 7 x 512   maxpooling with convolution.....
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 192, 3, 3,
                          strides=(2, 2), border_mode='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3,
                            strides=(2, 2), border_mode='valid')

    branch_pool = tf.keras.layers.AveragePooling2D((3, 3), strides=(2, 2))(x)
    branch_pool = conv2d_bn(branch_pool, 128, 1, 1)
    
    x = tf.keras.layers.concatenate([branch3x3, branch7x7x3, branch_pool],
              axis=channel_axis,name='mixed8')

    # mixed 9: 8 x 8 x 512

    branch1x1 = Rec_conv2d_bn(x, 192, 1, 1)

    branch3x3 = Rec_conv2d_bn(x, 128, 1, 1)
    branch3x3_1 = Rec_conv2d_bn(branch3x3, 64, 1, 3)
    branch3x3_2 = Rec_conv2d_bn(branch3x3, 64, 3, 1)

    branch3x3 = tf.keras.layers.concatenate([branch3x3_1, branch3x3_2], axis=channel_axis)
    branch3x3 = tf.keras.layers.Activation('relu')(branch3x3)
    
    branch3x3dbl = Rec_conv2d_bn(x, 256, 1, 1)
    branch3x3dbl = Rec_conv2d_bn(branch3x3dbl, 128, 3, 3)
    branch3x3dbl_1 = Rec_conv2d_bn(branch3x3dbl, 128, 1, 3)
    branch3x3dbl_2 = Rec_conv2d_bn(branch3x3dbl, 128, 3, 1)

    branch3x3_ls = tf.keras.layers.add([branch3x3dbl_1, branch3x3dbl_2])
    branch3x3dbl_ls = tf.keras.layers.Activation('relu')(branch3x3_ls)
    
    branch_pool = tf.keras.layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    IR_out3 = tf.keras.layers.concatenate([branch1x1, branch3x3, branch3x3dbl_ls, branch_pool],
                  axis=channel_axis)

    x = tf.keras.layers.add([x, IR_out3])
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(nm_classes, activation='softmax', name='predictions')(x)
    # Create model
    
    model = tf.keras.Model(inputs=[img_input], outputs=[x])
        
    #model = Model(img_input, x)
    
    return model