# -*- coding: utf-8 -*-
'''
Codes for small scale implementation of IRRCNN model by Md Zahangir Alom
# Reference:
- [Improved Inception-Residual Convolutional Neural Network for Object Recognition](https://arxiv.org/abs/1712.09888)
'''
from __future__ import print_function
from keras.models import Model
from keras import layers
from keras.layers import Flatten, Dense, Activation, Input, BatchNormalization,add, Dropout
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras import backend as K
from keras.regularizers import l2

#input_shape = (3,128,128)

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
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
 
    x = Conv2D(nb_filter, (nb_row, nb_col),strides=strides,
                      padding=border_mode,
                      name=conv_name,kernel_regularizer=l2(0.0002),data_format='channels_last')(x)
    x=Activation('relu')(x) 
    x = BatchNormalization(axis=bn_axis, name=bn_name)(x)
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
                      padding='same',
                      name=conv_name,kernel_regularizer=l2(0.0002),data_format='channels_last')(x12) 
    x3=Activation('relu')(x3)      
    x13 = add([x1, x3])

    x4 = Conv2D(nb_filter, (nb_row, nb_col),strides=strides,
                      padding='same',
                      name=conv_name,kernel_regularizer=l2(0.0002),data_format='channels_last')(x13)
    x4=Activation('relu')(x4) 
   
    x14 = add([x1, x4])
    x=Activation('relu')(x14)
    
    x = BatchNormalization(axis=bn_axis, name=bn_name)(x14)
       
    return x



def build_IRRCNN(input_shape,nm_classes):
    
    if K.image_dim_ordering() == 'tf':
        channel_axis = -1
    else:
        channel_axis = 1
    
    img_input = Input(shape=input_shape)
     
    x = conv2d_bn(img_input, 32, 3, 3, border_mode='valid',)
    #x = conv2d_bn(x, 48, 3, 3, strides=(2, 2), border_mode='valid')
    x = conv2d_bn(x, 48, 3, 3, strides=(1, 1), border_mode='valid')
    x = conv2d_bn(x, 64, 3, 3)
    #x = conv2d_bn(x, 32, 3, 3, strides=(2, 2), border_mode='valid')
    #x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, border_mode='valid')  
    x = conv2d_bn(x, 192, 3, 3, border_mode='valid')
    #x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = conv2d_bn(x, 192, 3, 3, strides=(2, 2), border_mode='valid')
    # mixed 0, 1, 2: 35 x 35 x 256   Inception RCNN block # 1
    #for i in range(3):
    branch1x1 = Rec_conv2d_bn(x, 32, 1, 1)

    branch5x5 = Rec_conv2d_bn(x, 48, 1, 1)
    branch5x5 = Rec_conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = Rec_conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = Rec_conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = Rec_conv2d_bn(branch3x3dbl, 64, 3, 3)

    branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = Rec_conv2d_bn(branch_pool, 32, 1, 1)
    IR_out1 = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                  axis=channel_axis,name='mixed0') 
    x = add([x, IR_out1])
    x = Activation('relu')(x)
    
    # Output size: 14x14x192  
    # End of first IRRCNN block.....   
    # mixed 3: 17 x 17 x 352
   
    branch3x3 = conv2d_bn(x, 128, 3, 3, strides=(2, 2), border_mode='valid')
    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3,
                             strides=(2, 2), border_mode='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    branch_pool = conv2d_bn(branch_pool, 128, 1, 1)
    x = layers.concatenate([branch3x3, branch3x3dbl, branch_pool],
              axis=channel_axis,
              name='mixed3')

#    # mixed 4: 14 x 14 x 352
#    branch1x1 = conv2d_bn(x, 192, 1, 1)
#
#    branch7x7 = conv2d_bn(x, 128, 1, 1)
#    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
#    branch7x7 = conv2d_bn(branch7x7, 128, 7, 1)
#
#    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
#    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
#    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
#    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
#    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
#
#    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
#    branch_pool = conv2d_bn(branch_pool, 128, 1, 1)
#    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
#              axis=channel_axis,
#              name='mixed4')

    # mixed 5, 6: 14 x 14 x 512
    # for i in range(2):
    branch1x1 = Rec_conv2d_bn(x, 128, 1, 1)

    branch7x7 = Rec_conv2d_bn(x, 160, 1, 1)
    branch7x7 = Rec_conv2d_bn(branch7x7, 160, 1, 7)
    branch7x7 = Rec_conv2d_bn(branch7x7, 64, 7, 1)

    branch7x7dbl = Rec_conv2d_bn(x, 160, 1, 1)
    branch7x7dbl = Rec_conv2d_bn(branch7x7dbl, 160, 7, 1)
    branch7x7dbl = Rec_conv2d_bn(branch7x7dbl, 160, 1, 7)
    branch7x7dbl = Rec_conv2d_bn(branch7x7dbl, 160, 7, 1)
    branch7x7dbl = Rec_conv2d_bn(branch7x7dbl, 64, 1, 7)

    branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = Rec_conv2d_bn(branch_pool, 96, 1, 1)
    IR_out2 = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                  axis=channel_axis)
    x = add([x, IR_out2])
    x = Activation('relu')(x)
    
    # End of second IRRCNN block.....
    
    # mixed 7: 14 x 14 x 512
    branch1x1 = conv2d_bn(x, 128, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 128, 7, 1)

    branch7x7dbl = conv2d_bn(x, 160, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 128, 1, 1)
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
              axis=channel_axis,name='mixed7')

    # mixed 8: 7 x 7 x 512
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 192, 3, 3,
                          strides=(2, 2), border_mode='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3,
                            strides=(2, 2), border_mode='valid')

    branch_pool = AveragePooling2D((3, 3), strides=(2, 2))(x)
    branch_pool = conv2d_bn(branch_pool, 128, 1, 1)
    
    x = layers.concatenate([branch3x3, branch7x7x3, branch_pool],
              axis=channel_axis,name='mixed8')

    # mixed 9: 8 x 8 x 512

    branch1x1 = Rec_conv2d_bn(x, 192, 1, 1)

    branch3x3 = Rec_conv2d_bn(x, 128, 1, 1)
    branch3x3_1 = Rec_conv2d_bn(branch3x3, 64, 1, 3)
    branch3x3_2 = Rec_conv2d_bn(branch3x3, 64, 3, 1)

    branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2], axis=channel_axis)
    branch3x3 = Activation('relu')(branch3x3)
    
    branch3x3dbl = Rec_conv2d_bn(x, 448, 1, 1)
    branch3x3dbl = Rec_conv2d_bn(branch3x3dbl, 128, 3, 3)
    branch3x3dbl_1 = Rec_conv2d_bn(branch3x3dbl, 128, 1, 3)
    branch3x3dbl_2 = Rec_conv2d_bn(branch3x3dbl, 128, 3, 1)
#    branch3x3dbl = merge([branch3x3dbl_1, branch3x3dbl_2],
#                             mode='concat', concat_axis=channel_axis)
    branch3x3_ls = add([branch3x3dbl_1, branch3x3dbl_2])
    branch3x3dbl_ls = Activation('relu')(branch3x3_ls)
    
    branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    IR_out3 = layers.concatenate([branch1x1, branch3x3, branch3x3dbl_ls, branch_pool],
                  axis=channel_axis)

    x = add([x, IR_out3])
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(nm_classes, activation='softmax', name='predictions')(x)
    # Create model
    model = Model(img_input, x)
    
    return model


def build_IRRCNN_V2_DFMPS(input_shape,nm_classes):

    if K.image_dim_ordering() == 'tf':
        channel_axis = -1
    else:
        channel_axis = 1

    img_input = Input(shape=input_shape)
    #x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), border_mode='valid')
    x = conv2d_bn(img_input, 64, 3, 3, border_mode='valid')
    x = conv2d_bn(x, 64, 3, 3, strides=(2, 2), border_mode='valid')
    x = conv2d_bn(x, 128, 3, 3)
    #x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 192, 1, 1, border_mode='valid')  
    x = conv2d_bn(x, 256, 3, 3, border_mode='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    # mixed 0, 1, 2: 35 x 35 x 256   Inception RCNN block # 1
    branch1x1 = Rec_conv2d_bn(x, 64, 1, 1)

    branch5x5 = Rec_conv2d_bn(x, 64, 1, 1)
    branch5x5 = Rec_conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = Rec_conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = Rec_conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = Rec_conv2d_bn(branch3x3dbl, 64, 3, 3)

    branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = Rec_conv2d_bn(branch_pool, 64, 1, 1)
    IR_out1 = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                  axis=channel_axis,name='mixed0') 
    x = add([x, IR_out1])
    x = Activation('relu')(x)
    
    # Output size: 14x14x192   
    # End of first IRRCNN block.....   
   
    branch3x3 = conv2d_bn(x, 256, 3, 3, strides=(2, 2), border_mode='valid')
    branch3x3dbl = conv2d_bn(x, 128, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 128, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 128, 3, 3,
                             strides=(2, 2), border_mode='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    branch_pool = conv2d_bn(branch_pool, 128, 1, 1)
    x = layers.concatenate([branch3x3, branch3x3dbl, branch_pool],
              axis=channel_axis,
              name='mixed3')

    # mixed 5, 6: 14 x 14 x 512
    # for i in range(2):
    branch1x1 = Rec_conv2d_bn(x, 128, 1, 1)

    branch7x7 = Rec_conv2d_bn(x, 160, 1, 1)
    branch7x7 = Rec_conv2d_bn(branch7x7, 160, 1, 7)
    branch7x7 = Rec_conv2d_bn(branch7x7, 128, 7, 1)

    branch7x7dbl = Rec_conv2d_bn(x, 160, 1, 1)
    branch7x7dbl = Rec_conv2d_bn(branch7x7dbl, 160, 7, 1)
    branch7x7dbl = Rec_conv2d_bn(branch7x7dbl, 160, 1, 7)
    branch7x7dbl = Rec_conv2d_bn(branch7x7dbl, 160, 7, 1)
    branch7x7dbl = Rec_conv2d_bn(branch7x7dbl, 128, 1, 7)

    branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = Rec_conv2d_bn(branch_pool, 128, 1, 1)
    IR_out2 = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                  axis=channel_axis)
    x = add([x, IR_out2])
    x = Activation('relu')(x)
    
    # End of second IRRCNN block.....    
    # mixed 7: 14 x 14 x 1024
    branch1x1 = conv2d_bn(x, 256, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 256, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 128, 1, 1)
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
              axis=channel_axis,name='mixed7')

    # mixed 8: 7 x 7 x 1024
    branch3x3 = conv2d_bn(x, 256, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 256, 3, 3,
                          strides=(2, 2), border_mode='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 256, 3, 3,
                            strides=(2, 2), border_mode='valid')

    branch_pool = AveragePooling2D((3, 3), strides=(2, 2))(x)
    branch_pool = conv2d_bn(branch_pool, 256, 1, 1)
    
    x = layers.concatenate([branch3x3, branch7x7x3, branch_pool],
              axis=channel_axis,name='mixed8')
              
    x = conv2d_bn(x, 1024, 1, 1)
    # mixed 9: 8 x 8 x 512

    branch1x1 = Rec_conv2d_bn(x, 256, 1, 1)

    branch3x3 = Rec_conv2d_bn(x, 128, 1, 1)
    branch3x3_1 = Rec_conv2d_bn(branch3x3, 128, 1, 3)
    branch3x3_2 = Rec_conv2d_bn(branch3x3, 128, 3, 1)

    branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2], axis=channel_axis)
    branch3x3 = Activation('relu')(branch3x3)
    
    branch3x3dbl = Rec_conv2d_bn(x, 512, 1, 1)
    branch3x3dbl = Rec_conv2d_bn(branch3x3dbl, 128, 3, 3)
    branch3x3dbl_1 = Rec_conv2d_bn(branch3x3dbl, 256, 1, 3)
    branch3x3dbl_2 = Rec_conv2d_bn(branch3x3dbl, 256, 3, 1)
#    branch3x3dbl = merge([branch3x3dbl_1, branch3x3dbl_2],
#                             mode='concat', concat_axis=channel_axis)
    branch3x3_ls = add([branch3x3dbl_1, branch3x3dbl_2])
    branch3x3dbl_ls = Activation('relu')(branch3x3_ls)
    
    branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 256, 1, 1)
    IR_out3 = layers.concatenate([branch1x1, branch3x3, branch3x3dbl_ls, branch_pool],
                  axis=channel_axis)

    x = add([x, IR_out3])
    
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(nm_classes, activation='softmax', name='predictions')(x)
    # Create model
    model = Model(img_input, x)
    
    return model
    

