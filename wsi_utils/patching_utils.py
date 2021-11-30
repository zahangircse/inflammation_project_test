# -*- coding: utf-8 -*-
"""
reated on Sat May  9 22:34:50 2020

@author: zahangir
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pandas as pd
import json
import glob
import random

from PIL import Image
import scipy.io as sio
import cv2
from os import listdir
from os.path import join as join_path
import pdb
from matplotlib import pyplot as plt

from collections import defaultdict
import csv
import shutil

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#from scipy import sp
import scipy.ndimage as ndimage
from skimage.transform import resize
import shutil

kernel = np.ones((5,5), np.uint8) 

IMG_CHANNELS = 3
patch_h = 128
patch_w = 128

img_rows = 256
img_cols = 256

IMG_HEIGHT, IMG_WIDTH = 2084,2084
number_samples_per_images = 200

abspath = os.path.dirname(os.path.abspath(__file__))
allowed_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp','*.mat','*.tif']


current_work_dir=os.getcwd()


def blue_ratio_image(image):
    
    height, width, channels = image.shape     
    BR_image = np.zeros((height, width),dtype=np.uint8)  #float32         
    for row in range(height):
        for column in range(width):
            pixel_values = np.squeeze(image[row,column,:])
            BR = ((100*pixel_values[0])/(1+pixel_values[1]+pixel_values[2]))*(256/(1+pixel_values[0]+pixel_values[1]+pixel_values[2]))    
            BR_image[row,column] = BR
           
    # apply laplacian of gaussian (LoG) filter on Blue ratio image..    
    #BR_image = laplace_of_gaussian(BR_image)
       
    return BR_image



def stain_normalization_OD(img,path_outputs):
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Io = 240 # Transmitted light intensity, Normalizing factor for image intensities
    alpha = 1  #As recommend in the paper. tolerance for the pseudo-min and pseudo-max (default: 1)
    beta = 0.15 #As recommended in the paper. OD threshold for transparent pixels (default: 0.15)
    ######## Step 1: Convert RGB to OD ###################
    ## reference H&E OD matrix.
    #Can be updated if you know the best values for your image. 
    #Otherwise use the following default values. 
    #Read the above referenced papers on this topic. 
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
    ### reference maximum stain concentrations for H&E
    maxCRef = np.array([1.9705, 1.0308])
    # extract the height, width and num of channels of image
    h, w, c = img.shape
    # reshape image to multiple rows and 3 columns.
    #Num of rows depends on the image size (wxh)
    img = img.reshape((-1,3))
    # calculate optical density
    # OD = −log10(I)  
    #OD = -np.log10(img+0.004)  #Use this when reading images with skimage
    #Adding 0.004 just to avoid log of zero. 
    OD = -np.log10((img.astype(np.float)+1)/Io) #Use this for opencv imread
    #Add 1 in case any pixels in the image have a value of 0 (log 0 is indeterminate)
    ############ Step 2: Remove data with OD intensity less than β ############
    # remove transparent pixels (clear region with no tissue)
    ODhat = OD[~np.any(OD < beta, axis=1)] #Returns an array where OD values are above beta
    #Check by printing ODhat.min()
    ############# Step 3: Calculate SVD on the OD tuples ######################
    #Estimate covariance matrix of ODhat (transposed)
    # and then compute eigen values & eigenvectors.
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    ######## Step 4: Create plane from the SVD directions with two largest values ######
    #project on the plane spanned by the eigenvectors corresponding to the two 
    # largest eigenvalues    
    That = ODhat.dot(eigvecs[:,1:3]) #Dot product
    ############### Step 5: Project data onto the plane, and normalize to unit length ###########
    ############## Step 6: Calculate angle of each point wrt the first SVD direction ########
    #find the min and max vectors and project back to OD space
    phi = np.arctan2(That[:,1],That[:,0])
    
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)
    
    vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    # a heuristic to make the vector corresponding to hematoxylin first and the 
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:    
        HE = np.array((vMin[:,0], vMax[:,0])).T
    else:
        HE = np.array((vMax[:,0], vMin[:,0])).T
    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T
    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE,Y, rcond=None)[0]
    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
    tmp = np.divide(maxC,maxCRef)
    C2 = np.divide(C,tmp[:, np.newaxis])
    ###### Step 8: Convert extreme values back to OD space
    # recreate the normalized image using reference mixing matrix 
    
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm>255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  
    plt.imsave(path_outputs, Inorm)



def extract_image_seq_non_overlapped_patches_and_masks(full_img,full_mask,patch_h,patch_w, img_name, imd_saving_dir):   
    
    height,width = full_mask.shape     
    rows = (int) (height/patch_h)
    columns = (int) (width/patch_w)
    k = 0
    pn = 0
    for r_s in range(rows):
        for c_s in range(columns):

            patch_img = full_img[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w,:]
            patch_mask = full_mask[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w]                
            f_img_name =str(img_name)+'_'+str(pn)+'.png'
            f_mask_name =str(img_name)+'_'+str(pn)+'_mask'+'.png'           
            final_des_img = os.path.join(imd_saving_dir,f_img_name)
            final_des_mask = os.path.join(imd_saving_dir,f_mask_name)            
            mx_val = patch_mask.max()
            mn_val = patch_mask.min()            
            print ('max_val : '+str(mx_val))
            print ('min_val : '+str(mn_val))            
            #cv2.imwrite(final_des_img,patch_img)
            #cv2.imwrite(final_des_mask,patch_mask)
            if mx_val > 0:
                cv2.imwrite(final_des_img,patch_img)
                cv2.imwrite(final_des_mask,patch_mask)

            pn+=1
            
        k +=1
        print ('Processing for: ' +str(k))
        
    return pn

def extract_image_seq_non_overlapped_patches(full_img,full_mask,patch_h,patch_w, img_name, imd_saving_dir):   
    
    height,width = full_mask.shape     
    rows = (int) (height/patch_h)
    columns = (int) (width/patch_w)
    k = 0
    pn = 0
    for r_s in range(rows):
        for c_s in range(columns):

            patch_img = full_img[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w,:]
            f_img_name =str(img_name)+'_'+str(pn)+'.png'
            final_des_img = os.path.join(imd_saving_dir,f_img_name)
            mx_val = patch_img.max()
            mn_val = patch_img.min()            
            print ('max_val : '+str(mx_val))
            print ('min_val : '+str(mn_val))            
            #if mx_val > 0:
            cv2.imwrite(final_des_img,patch_img)

            pn+=1           
        k +=1
        print ('Processing for: ' +str(k))
        
    return pn

def extract_image_seq_non_overlapped_patches_blue_ratio_normalized(HPF_img,patch_h,patch_w, img_name, imd_saving_dir):   
    
    height,width,channels = HPF_img.shape     
    rows = (int) (height/patch_h)
    columns = (int) (width/patch_w)
    k = 0
    pn = 0
    for r_s in range(rows):
        for c_s in range(columns):
            patch_img = HPF_img[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w,:]
            
            # apply blue ratio and define the threshold....
            m_blue_ratio_image = blue_ratio_image(patch_img)           
            max_val = m_blue_ratio_image.max()
            min_val = m_blue_ratio_image.min()
            avg_val = m_blue_ratio_image.mean()
            std_val = m_blue_ratio_image.std()
           
            print('\n Values for blue ratio regions')       
            print('maximum_value=', max_val)
            print('manimum_value=', min_val)
            print('mean_value=', avg_val)
            print('std_value=', std_val)
            
            threshold = avg_val  
            binary_image = 1.0 * (m_blue_ratio_image > threshold)
            pred = binary_image
            pred_mask = cv2.dilate(pred,kernel,iterations = 1)            
            pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)    #  To fillup the internal pixels...
            pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)     # opening operation to remove the noise 
            pred_mask = cv2.erode(pred_mask,kernel,iterations = 1)    
            morph_pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
                             
            white_pixel_cnt = cv2.countNonZero(morph_pred_mask)                       
            total_pixel = float(patch_h*patch_w)
            percent_wpxls = white_pixel_cnt/total_pixel
            
            #pdb.set_trace()
            print('Total white pixels:'+str(white_pixel_cnt))
            print('Percent of pixels ',percent_wpxls)
            
            #pdb.set_trace()
            f_img_name =str(img_name)+'_'+str(pn)+'.png'
            final_des_img = os.path.join(imd_saving_dir,f_img_name)
            #if white_pixel_cnt > ((patch_h*patch_w) * 0.20):
            #    cv2.imwrite(final_des_img,patch_img)

            low_th = int((patch_h*patch_w)*0.35)
            high_th = int((patch_h*patch_w)*0.9)
            if max_val>50 and (low_th<white_pixel_cnt and white_pixel_cnt < high_th):
                stain_normalization_OD(patch_img,final_des_img)

            pn+=1           
        k +=1
        print ('Processing for: ' +str(k))
        
    return pn

def data_aug_and_save_for_classification(full_imgs,img_name,num_agu_per_sample, img_saving_dir):
    
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    img = full_imgs  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=img_saving_dir, save_prefix=img_name, save_format='jpg'):
        i += 1
        if i > num_agu_per_sample:
            break  # otherwise the generator would loop indefinitely
            
def extract_image_random_patches(full_imgs,mask_img,central_xy, N_patches, patch_h,patch_w, img_name, imd_saving_dir):
       
    central_xy.astype(int)    
    height, width, chan = full_imgs.shape   
    start_x = patch_h/2
    start_y = patch_w/2
    end_x = height-start_x
    end_y = width - start_y
    
    k=0
    pn = 0
    while k <N_patches:
        x_center = int(central_xy[k,0])
        y_center =  int(central_xy[k,1])
        
        if (x_center > start_x and y_center > start_y and x_center < end_x and y_center < end_y) :
            img_patch = full_imgs[y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2),:]
            mask_patch = mask_img[y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]

            f_img_name =str(img_name)+'_'+str(pn)+'.jpg'
            f_mask_name =str(img_name)+'_'+str(pn)+'_mask.jpg'

            final_img_des = os.path.join(imd_saving_dir,f_img_name)
            final_mask_des = os.path.join(imd_saving_dir,f_mask_name)
            

            mx_val = final_mask_des.max()
            mn_val = final_mask_des.min()
            
            print ('max_val : '+str(mx_val))
            print ('min_val : '+str(mn_val))
            #if mx_val > 10:
            #    cv2.imwrite(final_img_des,img_patch)
            #    cv2.imwrite(final_mask_des,mask_patch)             
            cv2.imwrite(final_img_des,img_patch)
            cv2.imwrite(final_mask_des,mask_patch)

            pn +=1
   
        k +=1  

    print ('Processing for: ' +str(k))
    
    return pn


def create_training_patches_and_mask_from_sub_dir(data_path, patch_h, patch_w, img_saving_dir):
    
    for path, subdirs, files in os.walk(data_path):        
        for name in subdirs:            
            sub_dir = os.path.join(path, name+'/')           
            print(name)            
            images = [x for x in sorted(os.listdir(sub_dir)) if x[-9:] == '_mask.jpg']            
            #pdb.set_trace()            
            for i, mask_name in enumerate(images):               
                acc_name = mask_name.split('.')[0]  
                img_name = acc_name.split('_m')[0]
                image = cv2.imread(sub_dir + img_name+'.jpg', cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')                
                mask_im = cv2.imread(os.path.join(sub_dir, mask_name), cv2.IMREAD_GRAYSCALE)

                num_patches = extract_image_seq_non_overlapped_patches_and_masks(image, mask_im, patch_h, patch_w, img_name, img_saving_dir)
                
                print ('Processing done for: ' +str(i))
                        

def create_training_patches_from_sub_dir_blue_ratio_normalized(data_path, patch_h, patch_w, img_saving_dir):
    
    for path, subdirs, files in os.walk(data_path):        
        for dir_name in subdirs:            
            sub_dir_path = os.path.join(path, dir_name+'/')           
            print(dir_name)    
            
            if not os.path.isdir("%s/%s"%(img_saving_dir,dir_name)):
                os.makedirs("%s/%s"%(img_saving_dir,dir_name))                
            final_img_saving_dir = join_path(img_saving_dir,dir_name+'/')
                 
            # Checkk the sampels and read images for sub-patching....
            images = [x for x in sorted(os.listdir(sub_dir_path)) if x[-9:] == '.jpg' or '.png' or 'jpeg' or '.tif']    
            
            for i, img_name in enumerate(images):               
                acc_name = img_name.split('.')[0]  
                img_ext = img_name.split('.')[1]
                input_image = cv2.imread(sub_dir_path + img_name, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')                                
                img_name = dir_name+'_'+acc_name
                num_patches = extract_image_seq_non_overlapped_patches_blue_ratio_normalized(input_image,patch_h,patch_w, img_name, final_img_saving_dir)
                
                print ('Processing done for: ' +str(i))


def create_training_patches_from_sub_sub_dir_blue_ratio_normalized(data_path, patch_h, patch_w, img_saving_dir):
                  
    for path, subdirs, files in os.walk(data_path):        
        for dir_name in subdirs:            
            sub_dir_path = os.path.join(data_path, dir_name+'/')           
            print(dir_name)    
            sub_dir = os.path.join(data_path, dir_name+'/')
            
            if not os.path.isdir("%s/%s"%(img_saving_dir,dir_name)):
                os.makedirs("%s/%s"%(img_saving_dir,dir_name))                
            final_img_saving_dir = join_path(img_saving_dir,dir_name+'/')
            
            for path_2, subdirs_2,files_2 in os.walk(sub_dir):               
                for dir_name_2 in subdirs_2:                    
                    sub_sub_dir = os.path.join(path_2, dir_name_2+'/')
                    # Checkk the sampels and read images for sub-patching....
                    images = [x for x in sorted(os.listdir(sub_sub_dir)) if x[-4:] == '.jpg' or '.png' or 'jpeg' or '.tif']   
                    #pdb.set_trace()
                    for i, img_name in enumerate(images):               
                        acc_name = img_name.split('.')[0]  
                        #img_ext = img_name.split('.')[1]
                        input_image = cv2.imread(sub_sub_dir + img_name, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')                                
                        img_name = dir_name+'_'+dir_name_2+'_'+acc_name
                        num_patches = extract_image_seq_non_overlapped_patches_blue_ratio_normalized(input_image,patch_h,patch_w, img_name, final_img_saving_dir)                        
                        print ('Processing done for: ' +str(i))

                        
def create_training_patches_and_mask_from_sub_dir_WhitePxCnt(data_path, patch_h, patch_w, img_saving_dir):
    
    for path, subdirs, files in os.walk(data_path):       
        for name in subdirs:            
            sub_dir = os.path.join(path, name+'/')           
            #print(name)  
            print ('Processing done for: ' +name)                       
            images = [x for x in sorted(os.listdir(sub_dir)) if x[-9:] == '_mask.png']     
            pn = 0
            for i, mask_name in enumerate(images):               
                acc_name = mask_name.split('.')[0]  
                img_name = acc_name.split('_m')[0]
                patch_img = cv2.imread(sub_dir + img_name+'.png', cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')                
                mask_im = cv2.imread(os.path.join(sub_dir, mask_name), cv2.IMREAD_GRAYSCALE)
                
                white_pixel_cnt = cv2.countNonZero(mask_im)           
                f_img_name =str(img_name)+'_'+str(pn)+'.png'
                final_des_img = os.path.join(img_saving_dir,f_img_name)

                if white_pixel_cnt > ((patch_h*patch_w) * 0.90):
                    cv2.imwrite(final_des_img,patch_img)
                    pn+=1           
                #num_patches = extract_image_seq_non_overlapped_patches (image, mask_im, patch_h, patch_w, img_name, img_saving_dir)


    
def create_training_samples_with_blue_ratio(data_path, patch_h, patch_w, img_saving_dir):
    
    images = [x for x in sorted(os.listdir(data_path)) if x[-4:] == '.png']

    pn = 0
            
    for i, image_name in enumerate(images):               
        acc_name = image_name.split('.')[0]  
        print('procssing for ' + str(acc_name))
        #img_name = acc_name.split('_m')[0]
        patch_img = cv2.imread(data_path + image_name, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')   

        m_blue_ratio_image = blue_ratio_image(patch_img)
        
        max_val = m_blue_ratio_image.max()
        min_val = m_blue_ratio_image.min()
        avg_val = m_blue_ratio_image.mean()
        std_val = m_blue_ratio_image.std()
       
        print('\n Values for mitosis region')       
        print('maximum_value=', max_val)
        print('manimum_value=', min_val)
        print('mean_value=', avg_val)
        print('std_value=', std_val)
        
        threshold = min_val

        binary_image = 1.0 * (m_blue_ratio_image > avg_val)
        pred = binary_image
        pred_mask = cv2.dilate(pred,kernel,iterations = 1)            
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)    #  To fillup the internal pixels...
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)     # opening operation to remove the noise 
        pred_mask = cv2.erode(pred_mask,kernel,iterations = 1)    
        morph_pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
                         
        white_pixel_cnt = cv2.countNonZero(morph_pred_mask)           
        final_des_img = os.path.join(img_saving_dir,image_name)
        
        total_pixel = float(patch_h*patch_w)
        percent_wpxls = white_pixel_cnt/total_pixel
        
        #pdb.set_trace()
        print('Total white pixels:'+str(white_pixel_cnt))
        print('Percent of pixels ',percent_wpxls)
        
        if white_pixel_cnt > ((patch_h*patch_w) * 0.90):
            cv2.imwrite(final_des_img,patch_img)
            pn+=1  
    
    print('Total number of patches: ',str(pn))

def create_training_images_79_to_110(data_path, patch_h, patch_w, img_saving_dir):
    
    images = [x for x in sorted(os.listdir(data_path)) if x[-9:] == '_mask.png']

    pn = 0
            
    for i, mask_name in enumerate(images):               
        acc_mask_name = mask_name.split('.')[0]  
        acc_image_name = acc_mask_name.split('_m')[0]
        image_name = acc_image_name+'.png'
        print('procssing for ' + str(acc_image_name))
        part_img_id = acc_image_name.split('or_')[1]
        img_id = acc_image_name.split('_')[-3]
        #pdb.set_trace()
        if (int(img_id)>=79):
            full_img = cv2.imread(data_path + image_name, cv2.IMREAD_UNCHANGED).astype('float32')   
            full_mask = cv2.imread(data_path + mask_name, cv2.IMREAD_GRAYSCALE)       
            extract_image_seq_non_overlapped_patches(full_img,full_mask,patch_h,patch_w, acc_image_name, img_saving_dir)


def create_training_patches_masks_with_blue_ratio(data_path, patch_h, patch_w, img_saving_dir):
    
    images = [x for x in sorted(os.listdir(data_path)) if x[-9:] == '_mask.png']

    pn = 0
            
    for i, mask_name in enumerate(images):               
        acc_mask_name = mask_name.split('.')[0]  
        acc_image_name = acc_mask_name.split('_m')[0]
        image_name = acc_image_name+'.png'
        print('procssing for ' + str(acc_image_name))
        #img_name = acc_name.split('_m')[0]
        patch_img = cv2.imread(data_path + image_name, cv2.IMREAD_UNCHANGED).astype('float32')   
        patch_mask = cv2.imread(data_path + mask_name, cv2.IMREAD_GRAYSCALE)
        
        m_blue_ratio_image = blue_ratio_image(patch_img)
        
        max_val = m_blue_ratio_image.max()
        min_val = m_blue_ratio_image.min()
        avg_val = m_blue_ratio_image.mean()
        std_val = m_blue_ratio_image.std()
       
        print('\n Values for mitosis region')       
        print('maximum_value=', max_val)
        print('manimum_value=', min_val)
        print('mean_value=', avg_val)
        print('std_value=', std_val)
        
        threshold = min_val

        binary_image = 1.0 * (m_blue_ratio_image > avg_val)
        pred = binary_image
        pred_mask = cv2.dilate(pred,kernel,iterations = 1)            
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)    #  To fillup the internal pixels...
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)     # opening operation to remove the noise 
        pred_mask = cv2.erode(pred_mask,kernel,iterations = 1)    
        morph_pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
                         
        white_pixel_cnt = cv2.countNonZero(morph_pred_mask)           
        final_des_img = os.path.join(img_saving_dir,image_name)
        final_des_mask = os.path.join(img_saving_dir,mask_name)
        
        total_pixel = float(patch_h*patch_w)
        percent_wpxls = white_pixel_cnt/total_pixel
        
        #pdb.set_trace()
        print('Total white pixels:'+str(white_pixel_cnt))
        print('Percent of pixels ',percent_wpxls)
        
        if white_pixel_cnt > ((patch_h*patch_w) * 0.35):
            cv2.imwrite(final_des_img,patch_img)
            cv2.imwrite(final_des_mask,patch_mask)

            pn+=1  
    
    print('Total number of patches: ',str(pn))


                  
def select_normal_class_from_sub_dir_HNE(data_path, patch_h, patch_w, img_saving_dir):
    
    for path, subdirs, files in os.walk(data_path):       
        for name in subdirs:            
            sub_dir = os.path.join(path, name+'/')           
            print ('Processing done for: ' +name)                       
            images = [x for x in sorted(os.listdir(sub_dir)) if x[-4:] == '.png']             
            num_samples = len(images)
            if num_samples > 0:
                pn = 0
                for i, acc_name in enumerate(images):               
                    #acc_name = mask_name.split('.')[0]  
                    #img_name = acc_name.split('_m')[0]
                    patch_img = cv2.imread(sub_dir + acc_name, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')                
                    #mask_im = cv2.imread(os.path.join(sub_dir, mask_name), cv2.IMREAD_GRAYSCALE)                
                    #white_pixel_cnt = cv2.countNonZero(mask_im)           
                    #f_img_name =str(img_name)+'_'+str(pn)+'.png'
                    final_des_img = os.path.join(img_saving_dir,acc_name)
                    #if white_pixel_cnt > ((patch_h*patch_w) * 0.90):
                    cv2.imwrite(final_des_img,patch_img)
                    pn+=1  

def create_training_tumor_class_from_dir(images_masks_path,patch_h, patch_w,img_saving_dir):
        
    #TP_images_path_final = glob.glob(osp.join(TP_image_path, '*.png'))
    #print ('Processing done for: ' +name)                       
    images = [x for x in sorted(os.listdir(images_masks_path)) if x[-9:] == '_mask.png']  
    
    pn = 0          
    for i, mask_name in enumerate(images):               
        acc_name = mask_name.split('.')[0]  
        img_name = acc_name.split('_m')[0]
        print ('Processing done for: ' +img_name)                       

        patch_img = cv2.imread(images_masks_path + img_name+'.png', cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')                
        mask_im = cv2.imread(os.path.join(images_masks_path, mask_name), cv2.IMREAD_GRAYSCALE)
                
        white_pixel_cnt = cv2.countNonZero(mask_im)           
        f_img_name =str(img_name)+'_'+str(pn)+'.png'
        final_des_img = os.path.join(img_saving_dir,f_img_name)
        if white_pixel_cnt > ((patch_h*patch_w) * 0.99):
            cv2.imwrite(final_des_img,patch_img)
            pn+=1                          


def extract_HPFs_from_roi_normal_tissue(full_img, patch_h, patch_w, img_name, img_saving_dir):
    

    height,width, channel = full_img.shape    
    rows = (int) (height/patch_h)
    columns = (int) (width/patch_w)    
    #pdb.set_trace()
    k = 0
    pn = 0
    for r_s in range(rows):
        for c_s in range(columns):
            patch_img = full_img[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w,:]            
            patch_hsv = cv2.cvtColor(patch_img, cv2.COLOR_BGR2HSV)
            lower_red = np.array([20, 20, 20])
            upper_red = np.array([200, 200, 200])
            mask_patch = cv2.inRange(patch_hsv, lower_red, upper_red)
            white_pixel_cnt = cv2.countNonZero(mask_patch)
           
            f_img_name =str(img_name)+'_'+str(pn)+'.png'
            final_des_img = os.path.join(img_saving_dir,f_img_name)

            #if white_pixel_cnt > ((patch_h*patch_w) * 0.50):
            cv2.imwrite(final_des_img,patch_img)
            pn+=1 
      
        k +=1
        print ('Processing for: ' +str(k))

    return pn

def randomly_selected_samples_256x256(data_path, patch_h, patch_w, number_of_samples, img_saving_dir):
    
    images = [x for x in sorted(os.listdir(data_path)) if x[-4:] == '.png']      
    total = len(images)            
   
    print('Total number of sampels: '+ str(total))   
    rand_sample_ids = random.sample(range(0,total),number_of_samples)

    print('Total random sampels: '+ str(rand_sample_ids))   

    for x in range(len(rand_sample_ids)):
        image_id = rand_sample_ids[x]
        print('Randomly selected idx: '+ str(image_id))             
        image_name = images[image_id]   
        img_name = image_name.split('.')[0]  
        patch_img = cv2.imread(data_path + img_name+'.png', cv2.IMREAD_UNCHANGED).astype('float32')                      
        f_img_name =str(img_name)+'.png'
        final_des_img = os.path.join(img_saving_dir,f_img_name)
        cv2.imwrite(final_des_img,patch_img)


def randomly_selected_samples_256x256_train_val(data_path, patch_h, patch_w, number_of_samples, img_saving_dir_train,img_saving_dir_val):
    
    images = [x for x in sorted(os.listdir(data_path)) if x[-4:] == '.png']      
    total = len(images)            
   
    print('Total number of sampels: '+ str(total))   
    rand_sample_ids = random.sample(range(0,total),number_of_samples)

    print('Total random sampels: '+ str(rand_sample_ids)) 
    
    counter = 0
    for x in range(len(rand_sample_ids)):
        image_id = rand_sample_ids[x]
        print('Randomly selected idx: '+ str(image_id))             
        image_name = images[image_id]   
        img_name = image_name.split('.')[0]  
        patch_img = cv2.imread(data_path + img_name+'.png', cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')                      
        f_img_name =str(img_name)+'.png'
        counter = counter+1
        
        if counter<50000:
            final_des_img = os.path.join(img_saving_dir_train,f_img_name)
            cv2.imwrite(final_des_img,patch_img)
        else:
            final_des_img = os.path.join(img_saving_dir_val,f_img_name)
            cv2.imwrite(final_des_img,patch_img)
            




def randomly_selected_images_mask_256x256(data_path, patch_h, patch_w, number_of_samples, img_saving_dir_train,img_saving_dir_val):
    
    images = [x for x in sorted(os.listdir(data_path)) if x[-9:] == '_mask.png']      
    total = len(images)            
   
    print('Total number of sampels: '+ str(total))   
    rand_sample_ids = random.sample(range(0,total),number_of_samples)
    print('Total random sampels: '+ str(rand_sample_ids))   
    
    counter = 0
    
    pdb.set_trace()
    
    for x in range(len(rand_sample_ids)):
        image_id = rand_sample_ids[x]
        print('Randomly selected idx: '+ str(image_id))             
        mask_name = images[image_id]   
        mask_name_woe = mask_name.split('.')[0] 
        image_name_woe = mask_name_woe.split('_m')[0]
        
        image = cv2.imread(data_path + image_name_woe+'.png', cv2.IMREAD_UNCHANGED).astype('float32')  
        mask_img = cv2.imread(os.path.join(data_path, mask_name), cv2.IMREAD_GRAYSCALE)
        
        counter = counter+1           
        f_img_name =str(image_name_woe)+'.png'
        
        if counter<=120000:
            final_des_img = os.path.join(img_saving_dir_train,f_img_name)        
            final_des_msk = os.path.join(img_saving_dir_train,mask_name)        
            cv2.imwrite(final_des_img,image)
            cv2.imwrite(final_des_msk,mask_img)
        else:
            final_des_img = os.path.join(img_saving_dir_val,f_img_name)        
            final_des_msk = os.path.join(img_saving_dir_val,mask_name)        
            cv2.imwrite(final_des_img,image)
            cv2.imwrite(final_des_msk,mask_img)
            
        
        

def randomly_crop_samples_64x64(data_path, patch_h, patch_w, number_of_samples, img_saving_dir):
    
    images = [x for x in sorted(os.listdir(data_path)) if x[-4:] == '.png']      
    total = len(images)            
   
    print('Total number of sampels: '+ str(total))   
    rand_sample_ids = random.sample(range(0,total),number_of_samples)
    print('Total random sampels: '+ str(rand_sample_ids))   

    for x in range(len(rand_sample_ids)):
        image_id = rand_sample_ids[x]
        print('Randomly selected idx: '+ str(image_id))             
        image_name = images[image_id]   
        img_name = image_name.split('.')[0]  
        patch_img = cv2.imread(data_path + img_name+'.png', cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')                      
        f_img_name =str(img_name)+'.png'
        final_des_img = os.path.join(img_saving_dir,f_img_name)
        height_rdm = 64
        weight_rdm = 64    
        start_x = random.randint(1,patch_h-height_rdm)   
        start_y = random.randint(1,patch_w-weight_rdm)
       
        child_patch = patch_img[start_y: start_y+height_rdm,start_x: start_x+weight_rdm,:]
        cv2.imwrite(final_des_img,child_patch)
        
        #second_patch for tumor
        f_img_name_1 =str(img_name)+'_1.png'
        final_des_img_1 = os.path.join(img_saving_dir,f_img_name_1)
        height_rdm = 64
        weight_rdm = 64    
        start_x = random.randint(1,patch_h-height_rdm)   
        start_y = random.randint(1,patch_w-weight_rdm)
       
        child_patch = patch_img[start_y: start_y+height_rdm,start_x: start_x+weight_rdm,:]
        cv2.imwrite(final_des_img_1,child_patch)


def create_training_normal_class_from_sub_dir(data_path, patch_h, patch_w, img_saving_dir):
        
    for path, subdirs, files in os.walk(data_path):       
        for name in subdirs:            
            sub_dir = os.path.join(path, name+'/')           
            print ('Processing done for: ' +name)                       
            images = [x for x in sorted(os.listdir(sub_dir)) if x[-4:] == '.jpg']      
            total = len(images)            
            #random_image_ids = random.randint(1,total)              
            #pdb.set_trace()     
            print('Total number of sampels: '+ str(total))
            
            if total > 20:
                for x in range(20):
                    image_id = random.randint(1,total-1)
                    print('Randomly selected idx: '+ str(image_id))             

                    image_name = images[image_id]   
                    img_name = image_name.split('.')[0]  
                    patch_img = cv2.imread(sub_dir + img_name+'.jpg', cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')                
                    num_patches = extract_HPFs_from_roi_normal_tissue (patch_img, patch_h, patch_w, img_name, img_saving_dir)
            else:
                
                 for i, mask_name in enumerate(images):   
                    acc_name = mask_name.split('.')[0]  
                    img_name = acc_name.split('_m')[0]
                    patch_img = cv2.imread(sub_dir + img_name+'.jpg', cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')                
                    num_patches = extract_HPFs_from_roi_normal_tissue (patch_img, patch_h, patch_w, img_name, img_saving_dir)

    
def create_training_data_from_sub_sub_dir(data_path, patch_h, patch_w, acc_name, img_saving_dir):
    
    for path, subdirs, files in os.walk(data_path):
        
        for name in subdirs:
            
            sub_dir = os.path.join(path, name+'/')
            
            print(name) 
            
            for path_2, subdirs_2,files_2 in os.walk(sub_dir):
                
                for name_2 in subdirs_2:
                    
                    sub_sub_dir = os.path.join(path_2, name_2+'/')
                    
                    images = os.listdir(sub_sub_dir)
                    
                    if name_2 =='0':
                        for image_name in images:  
                            read_img = cv2.imread(os.path.join(sub_sub_dir,image_name))
                            h,w,c=read_img.shape
                            if h==w:
                                shutil.copy(os.path.join(sub_sub_dir,image_name),os.path.join(img_saving_dir+'0/'))
                    else:
                        for image_name in images:                           
                            read_img = cv2.imread(os.path.join(sub_sub_dir,image_name))
                            h,w,c=read_img.shape
                            if h==w:
                                shutil.copy(os.path.join(sub_sub_dir,image_name),os.path.join(img_saving_dir+'1/'))
                    
                            

def create_patches_masks_from_dir(IMAGE_PATH,patch_h, patch_w, acc_name, img_saving_dir):     
    
    train_data_path = os.path.join(IMAGE_PATH)
    images = filter((lambda image: 'mask' not in image), os.listdir(train_data_path))   
    total = np.round(len(images))    
    i = 0
    print('Creating training images...')
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.jpg'       
        acc_name = image_name.split('.')[0]
        #img = cv2.imread(os.path.join(train_data_path, image_name))
        im = cv2.imread(os.path.join(train_data_path, image_name),cv2.IMREAD_UNCHANGED)
        mask_im = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
        img_rz = im
        img_mask_rz = mask_im      
        num_patches = extract_image_seq_non_overlapped_patches (img_rz, img_mask_rz, patch_h, patch_w, acc_name, img_saving_dir)       
        print ('Processing for: ' +str(i))
        #Total_patches = Total_patches + num_patches
    
    return 0

def create_dataset_random_patches_masks_driver(base_dir,patch_h,patch_w,data_saving_dir):
    
    train_data_path = os.path.join(base_dir)
    images = filter((lambda image: '_anno' not in image), os.listdir(train_data_path))
    total = np.round(len(images)) 

    pdb.set_trace()
    
    for filename in images:

        #print(filename)     
        img = cv2.imread(os.path.join(base_dir,filename))            
        #img_path_first= os.path.dirname (filename)
            #img_path_first =filename.split('.')[0]
        img_name = filename.split('/')[-1]
        img_name = img_name.split('.')[0]
        mask_name = img_name+'_anno.bmp'
            #mask_name = '/'+mask_name
        mask_path = os.path.join(base_dir,mask_name)           
        mask_img = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
        
        # show heatmaps image..
        #plt.imshow(mask_img, cmap='hot', interpolation='nearest')
        #plt.show() 
        #mask_img = 255*(mask_img[:,:]>0)
                    
        height, width, chan = img.shape           
        radius = min((height, width))           
        central_xy = np.random.random((number_samples_per_images, 2))*radius 
            
        central_xy=central_xy.astype(int)
           # print(class_index)
        print(img_name)
                        
        img_saving_dir_b = os.path.join(data_saving_dir,'images_and_masks_benign_malignant/benign/')  
            #img_saving_dir_m = os.path.join(data_dir,'images_and_masks_benign_malignant/malignant/') 

            #Extract random patches from image for each 
        if len(central_xy) > 0:
               num_patches = extract_image_random_patches (img, mask_img, central_xy, len(central_xy), patch_h, patch_w, img_name, img_saving_dir_b)

    return 0

def segmentation_images_masks_refinement(images_masks_path,TP_image_path,image_mask_saving_dir):
    
    
    TP_images_path_final = glob.glob(os.path.join(TP_image_path, '*.png'))
    TP_images_path_final.sort()

    train_data_path = os.path.join(images_masks_path)
    images = filter((lambda image: '_mask' not in image), os.listdir(train_data_path))   
    total = np.round(len(images))    
    i = 0
    print('Creating refined images and masks....')
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.png'       
        acc_name = image_name.split('.')[0]

        patch_img = cv2.imread(os.path.join(train_data_path, image_name),cv2.IMREAD_UNCHANGED)
        patch_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
        
        mx_val = patch_mask.max()
        mn_val = patch_mask.min()            
        print ('max_val : '+str(mx_val))
        print ('min_val : '+str(mn_val))            
      
        final_des_img = os.path.join(image_mask_saving_dir,image_name)
        final_des_mask = os.path.join(image_mask_saving_dir,image_mask_name) 
        
        if mx_val == 0:
            
            cv2.imwrite(final_des_img,patch_img)
            cv2.imwrite(final_des_mask,patch_mask)
    
    height, width, chan = patch_img.shape           
    radius = min((height, width))           
    central_xy = np.random.random((number_samples_per_images, 2))*radius 
    


def create_seg_mask_for_normal_images(normal_tissue_dir, TP_image_path, image_mask_saving_dir):   
    
    TP_images_path_final = glob.glob(os.path.join(TP_image_path, '*.png'))
    TP_images_path_final.sort()
    num_tp_samples = np.round(len(TP_images_path_final))
    
    images = [x for x in sorted(os.listdir(normal_tissue_dir)) if x[-4:] == '.png']
    total_samples = np.round(len(images))    
    i = 0
    print('Creating refined images and masks....')
    for image_name in images:

        image_mask_name = image_name.split('.')[0] + '_mask.png'       
        acc_name = image_name.split('.')[0]

        acc_im = cv2.imread(os.path.join(normal_tissue_dir, image_name),cv2.IMREAD_UNCHANGED)
        mask_h,mask_w,c=acc_im.shape
        mask_initial = np.zeros((mask_h,mask_w), dtype='float32')
        
        image_id_tp = random.random(1,num_tp_samples)    
        tp_img_path = TP_images_path_final[image_id_tp] 
        tp_im = cv2.imread(os.path.join(tp_img_path),cv2.IMREAD_UNCHANGED)
        
        height_rdm = random.randint(16,64)
        weight_rdm = random.randint(16,64)
    
        start_x = random.randint(1,mask_h-height_rdm)   
        start_y = random.randint(1,mask_w-weight_rdm)
        
        acc_im[start_y: start_y+height_rdm,start_x: start_x+weight_rdm,:] = tp_im[start_y: start_y+height_rdm,start_x: start_x+weight_rdm,:]
        child_mask = 255.0* np.ones((height_rdm,weight_rdm), dtype='float32')
        mask_initial[start_y: start_y+height_rdm,start_x: start_x+weight_rdm] = child_mask
          
        mx_val = mask_initial.max()
        mn_val = mask_initial.min()            
        print ('max_val : '+str(mx_val))
        print ('min_val : '+str(mn_val)) 

        final_des_img = os.path.join(image_mask_saving_dir,image_name)
        final_des_mask = os.path.join(image_mask_saving_dir,image_mask_name)                
      
        if mx_val > 0:
            cv2.imwrite(final_des_img,acc_im)
            cv2.imwrite(final_des_mask,mask_initial)
      

def extract_image_seq_non_overlapped_patches_WSG34(full_img,patch_h,patch_w, img_name, imd_saving_dir):   
    
    height,width,clrs = full_img.shape     
    rows = (int) (height/patch_h)
    columns = (int) (width/patch_w)
    k = 0
    pn = 0
    #pdb.set_trace()

    for r_s in range(rows):
        for c_s in range(columns):

            patch_img = full_img[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w,:]
            f_img_name =str(img_name)+'_'+str(pn)+'.png'
            final_des_img = os.path.join(imd_saving_dir,f_img_name)
            cv2.imwrite(final_des_img,patch_img)

            pn+=1
            
        k +=1
        print ('Processing for: ' +str(k))
        
    return pn

def create_test_database_WSG34(data_path, patch_h, patch_w, img_saving_dir):
        
    for path, subdirs, files in os.walk(data_path):       
        for name in subdirs:            
            sub_dir = os.path.join(path, name+'/')           
            print ('Processing done for: ' +sub_dir)                       
            images = [x for x in sorted(os.listdir(sub_dir)) if x[-4:] == '.jpg' or '.tif' or '.png']      
            total = len(images) 
             
            # creat directory for saving 256x samples...
            #pdb.set_trace()
            if not os.path.isdir("%s/%s"%(img_saving_dir,name)):
                os.makedirs("%s/%s"%(img_saving_dir,name))
	    # create all necessary path for saving log files
            img_saving_dir_final = join_path(img_saving_dir,name)            
            print(img_saving_dir_final+'/')
            #random_image_ids = random.randint(1,total)              
            #pdb.set_trace()     
            for path2, subdirs2, files2 in os.walk(sub_dir):
                for name2 in subdirs2:
                    img_dir = os.path.join(sub_dir,name2)
                    img_dir = img_dir+'/' 
                    images = [x for x in sorted(os.listdir(img_dir)) if x[-4:] == '.jpg' or '.tif' or '.png']      
                    total = len(images) 
                    print('Total number of sampels: '+ str(total))
            
                    #if total > 20:
                    for image_id in range(total):
                        #image_id = random.randint(1,total-1)
                        print('Randomly selected idx: '+ str(image_id))             
                        image_name = images[image_id]   
                        img_name = image_name.split('.')[0]
                        img_name_final = name2+'_'+img_name 
                        #pdb.set_trace() 
                        patch_img = cv2.imread(img_dir + img_name+'.tif', cv2.IMREAD_UNCHANGED)                
                        num_patches = extract_image_seq_non_overlapped_patches_WSG34(patch_img,patch_h,patch_w, img_name_final, img_saving_dir_final)   



def create_train_val_database_WSG34_NMT20(data_path, patch_h, patch_w, img_saving_dir):
        
    for path, subdirs, files in os.walk(data_path):       
        for name in subdirs:            
            sub_dir = os.path.join(path, name+'/')           
            print ('Processing done for: ' +sub_dir)                       
            images = [x for x in sorted(os.listdir(sub_dir)) if x[-4:] == '.jpg' or '.tif' or '.png']      
            total = len(images) 
             
            # creat directory for saving 256x samples...
            #pdb.set_trace()
            if not os.path.isdir("%s/%s"%(img_saving_dir,name)):
                os.makedirs("%s/%s"%(img_saving_dir,name))
	    # create all necessary path for saving log files
            img_saving_dir_final = join_path(img_saving_dir,name)            
            print(img_saving_dir_final+'/')
            #random_image_ids = random.randint(1,total)              
            #pdb.set_trace()     
            for path2, subdirs2, files2 in os.walk(sub_dir):
                for name2 in subdirs2:
                    img_dir = os.path.join(sub_dir,name2)
                    img_dir = img_dir+'/' 
                    images = [x for x in sorted(os.listdir(img_dir)) if x[-4:] == '.jpg' or '.tif' or '.png']      
                    total = len(images) 
                    print('Total number of sampels: '+ str(total))
                    
                    # consider 20 HPFs if folder contain morethan 20 HPFs
                    number_of_RDM_HPFs = 20

                    if total > 20:
                        rand_sample_ids = random.sample(range(0,total),number_of_RDM_HPFs)
                        print('Total random sampels: '+ str(rand_sample_ids))   
                        for image_id in range(len(rand_sample_ids)):
                            #image_id = random.randint(1,total-1)
                            print('Randomly selected idx: '+ str(image_id))             
                            image_name = images[image_id]   
                            img_name = image_name.split('.')[0]
                            img_name_final = name2+'_'+img_name 
                            #pdb.set_trace() 
                            patch_img = cv2.imread(img_dir + img_name+'.tif', cv2.IMREAD_UNCHANGED)                
                            num_patches = extract_image_seq_non_overlapped_patches_WSG34(patch_img,patch_h,patch_w, img_name_final, img_saving_dir_final)   
                    else:
                        for image_id in range(total):
                            #image_id = random.randint(1,total-1)
                            print('Randomly selected idx: '+ str(image_id))             
                            image_name = images[image_id]   
                            img_name = image_name.split('.')[0]
                            img_name_final = name2+'_'+img_name 
                            #pdb.set_trace() 
                            patch_img = cv2.imread(img_dir + img_name+'.tif', cv2.IMREAD_UNCHANGED)                
                            num_patches = extract_image_seq_non_overlapped_patches_WSG34(patch_img,patch_h,patch_w, img_name_final, img_saving_dir_final)   

               

def create_train_val_database_WSG34_NMT20_SCD(data_path, patch_h, patch_w, img_saving_dir):
        
    for path, subdirs, files in os.walk(data_path):       
        for name in subdirs:            
            sub_dir = os.path.join(path, name+'/')           
            print ('Processing done for: ' +sub_dir)                       
            images = [x for x in sorted(os.listdir(sub_dir)) if x[-4:] == '.jpg' or '.tif' or '.png']      
            total = len(images) 
             
            for path2, subdirs2, files2 in os.walk(sub_dir):
                for name2 in subdirs2:
                    img_dir = os.path.join(sub_dir,name2)
                    img_dir = img_dir+'/' 
                    images = [x for x in sorted(os.listdir(img_dir)) if x[-4:] == '.jpg' or '.tif' or '.png']      
                    total = len(images) 
                    print('Total number of sampels: '+ str(total))
                    
                    # consider 20 HPFs if folder contain morethan 20 HPFs
                    number_of_RDM_HPFs = 20

                    if total > 20:
                        rand_sample_ids = random.sample(range(0,total),number_of_RDM_HPFs)
                        print('Total random sampels: '+ str(rand_sample_ids))   
                        for image_id in range(len(rand_sample_ids)):
                            #image_id = random.randint(1,total-1)
                            print('Randomly selected idx: '+ str(image_id))             
                            image_name = images[image_id]   
                            img_name = image_name.split('.')[0]
                            img_name_final = name2+'_'+img_name 
                            #pdb.set_trace() 
                            patch_img = cv2.imread(img_dir + img_name+'.tif', cv2.IMREAD_UNCHANGED)                
                            num_patches = extract_image_seq_non_overlapped_patches_WSG34(patch_img,patch_h,patch_w, img_name_final, img_saving_dir)   
                    else:
                        for image_id in range(total):
                            #image_id = random.randint(1,total-1)
                            print('Randomly selected idx: '+ str(image_id))             
                            image_name = images[image_id]   
                            img_name = image_name.split('.')[0]
                            img_name_final = name2+'_'+img_name 
                            #pdb.set_trace() 
                            patch_img = cv2.imread(img_dir + img_name+'.tif', cv2.IMREAD_UNCHANGED)                
                            num_patches = extract_image_seq_non_overlapped_patches_WSG34(patch_img,patch_h,patch_w, img_name_final, img_saving_dir)   

               


def randomly_selected_samples_256x256(data_path, patch_h, patch_w, number_of_samples, img_saving_dir):
    
    images = [x for x in sorted(os.listdir(data_path)) if x[-4:] == '.png']      
    total = len(images)            
   
    print('Total number of sampels: '+ str(total))   
    rand_sample_ids = random.sample(range(0,total),number_of_samples)

    print('Total random sampels: '+ str(rand_sample_ids))   

    for x in range(len(rand_sample_ids)):
        image_id = rand_sample_ids[x]
        print('Randomly selected idx: '+ str(image_id))             
        image_name = images[image_id]   
        img_name = image_name.split('.')[0]  
        patch_img = cv2.imread(data_path + img_name+'.png', cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')                      
        f_img_name =str(img_name)+'.png'
        final_des_img = os.path.join(img_saving_dir,f_img_name)
        cv2.imwrite(final_des_img,patch_img)


def randomly_selected_samples_256x256_train_val(data_path, patch_h, patch_w, number_of_samples, img_saving_dir_train,img_saving_dir_val):
        
    for path2, subdirs2, files2 in os.walk(data_path):
        for name in subdirs2:
            img_dir = os.path.join(data_path,name)
            img_dir = img_dir+'/' 
            images = [x for x in sorted(os.listdir(img_dir)) if x[-4:] == '.jpg' or '.tif' or '.png']      
            total = len(images) 
            print('Total number of sampels: '+ str(total))
            rand_sample_ids = random.sample(range(0,total),number_of_samples)
            print('Total random sampels: '+ str(rand_sample_ids)) 
            
            if not os.path.isdir("%s/%s"%(img_saving_dir_train,name)):
                os.makedirs("%s/%s"%(img_saving_dir_train,name))
	    # create all necessary path for saving log files
            img_saving_dir_train_final = join_path(img_saving_dir_train,name)            
            print(img_saving_dir_train_final+'/')
           
            
            if not os.path.isdir("%s/%s"%(img_saving_dir_val,name)):
                os.makedirs("%s/%s"%(img_saving_dir_val,name))
	    # create all necessary path for saving log files
            img_saving_dir_val_final = join_path(img_saving_dir_val,name)            
            print(img_saving_dir_val_final+'/')
            #pdb.set_trace()
            counter = 0
            for x in range(number_of_samples):
                image_id = rand_sample_ids[x]
                print('Randomly selected idx: '+ str(image_id))             
                image_name = images[image_id]   
                img_name = image_name.split('.')[0]  
                patch_img = cv2.imread(img_dir + img_name+'.png', cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')                      
                f_img_name =str(img_name)+'.png'
        
                if counter<= 120000:
                   if counter < 100000:
                      final_des_img = os.path.join(img_saving_dir_train_final,f_img_name)
                      cv2.imwrite(final_des_img,patch_img)
                   else:
                      final_des_img = os.path.join(img_saving_dir_val_final,f_img_name)
                      cv2.imwrite(final_des_img,patch_img)
                counter = counter + 1
    return 0




def stain_norm_and_save_train_or_val_sub_dir(data_path, patch_h, patch_w, img_saving_dir):
        
    for path2, subdirs2, files2 in os.walk(data_path):
        for name in subdirs2:
            img_dir = os.path.join(data_path,name)
            img_dir = img_dir+'/' 
            images = [x for x in sorted(os.listdir(img_dir)) if x[-4:] == '.jpg' or '.tif' or '.png']      
            total = len(images) 
            print('Total number of sampels: '+ str(total))
            #rand_sample_ids = random.sample(range(0,total),number_of_samples)
            #print('Total random sampels: '+ str(rand_sample_ids)) 
            
            if not os.path.isdir("%s/%s"%(img_saving_dir,name)):
                os.makedirs("%s/%s"%(img_saving_dir,name))
	    # create all necessary path for saving log files
            img_saving_dir_final = join_path(img_saving_dir,name)            
            print(img_saving_dir_final+'/') 
            
            counter = 0
            for image_id in range(total):
                #image_id = images[x]
                print('Randomly selected idx: '+ str(image_id))             
                image_name = images[image_id]   
                img_name = image_name.split('.')[0]  
                patch_img = cv2.imread(img_dir + img_name+'.png', cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')                      
                img_rz = cv2.resize(patch_img, (patch_h, patch_w),interpolation = cv2.INTER_NEAREST) 
                m_blue_ratio_image = blue_ratio_image(img_rz)
                avg_val = m_blue_ratio_image.mean()       
                mx_val = m_blue_ratio_image.max()
                print('mean_value=', avg_val)
                binary_image = 1.0 * (m_blue_ratio_image > avg_val)
                pred = binary_image
                pred_mask = cv2.dilate(pred,kernel,iterations = 1)            
                pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)    #  To fillup the internal pixels...
                pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)     # opening operation to remove the noise 
                pred_mask = cv2.erode(pred_mask,kernel,iterations = 1)    
                morph_pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
                morph_pred_mask = morph_pred_mask*255.0 
                #mask_sd = os.path.join(img_saving_dir_final,img_name+'_mask.png')
                #cv2.imwrite(mask_sd,morph_pred_mask)
                white_pixel_cnt = int(cv2.countNonZero(morph_pred_mask))           
                #total_pixel = float(patch_h*patch_w)
                #pdb.set_trace()
                f_img_name =str(img_name)+'.png'
                print('Working for the image:'+str(f_img_name)) 
                final_des_img = os.path.join(img_saving_dir_final,f_img_name)
                low_th = int((patch_h*patch_w)*0.35)
                high_th = int((patch_h*patch_w)*0.9)
                if mx_val>50 and (low_th<white_pixel_cnt and white_pixel_cnt < high_th):
                   stain_normalization_OD(img_rz,final_des_img) 
                counter = counter + 1
    return 0




def stain_norm_and_save_train_or_val_dir(data_path, patch_h, patch_w, img_saving_dir_final):
        
    images = [x for x in sorted(os.listdir(data_path)) if x[-4:] == '.jpg' or '.tif' or '.png']      
    total = len(images) 
    print('Total number of sampels: '+ str(total))
    #img_saving_dir_final = join_path(img_saving_dir,name)            
    print(img_saving_dir_final)         
    #pdb.set_trace() 
    counter = 0
    for image_id in range(total):
        #image_id = images[x]
        print('Randomly selected idx: '+ str(image_id))             
        image_name = images[image_id]   
        img_name = image_name.split('.')[0]  
        patch_img = cv2.imread(data_path + img_name+'.png', cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')                      
        
        #pdb.set_trace()        
        img_rz = cv2.resize(patch_img, (patch_h, patch_w),interpolation = cv2.INTER_NEAREST) 
        m_blue_ratio_image = blue_ratio_image(img_rz)
        avg_val = m_blue_ratio_image.mean()       
        mx_val = m_blue_ratio_image.max()
        mn_val = m_blue_ratio_image.min()
        print('mean_value=', avg_val)
        print('Mx value :',mx_val)
        print('Mn value : ',mn_val)

        binary_image = 1.0 * (m_blue_ratio_image > avg_val)
        pred = binary_image
        pred_mask = cv2.dilate(pred,kernel,iterations = 1)            
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)    #  To fillup the internal pixels...
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)     # opening operation to remove the noise 
        pred_mask = cv2.erode(pred_mask,kernel,iterations = 1)    
        morph_pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
        morph_pred_mask = morph_pred_mask*255.0 
        #mask_sd = os.path.join(img_saving_dir_final,img_name+'_mask.png')
        #cv2.imwrite(mask_sd,morph_pred_mask)
        white_pixel_cnt = int(cv2.countNonZero(morph_pred_mask))           
        total_pixel = float(patch_h*patch_w)
        #pdb.set_trace()
        f_img_name =str(img_name)+'.png'
        print('Working for the image:'+str(f_img_name)) 
        final_des_img = os.path.join(img_saving_dir_final,f_img_name)
        low_th = int((patch_h*patch_w)*0.45)
        high_th = int((patch_h*patch_w)*0.85)
        if mx_val>50 and (low_th<white_pixel_cnt and white_pixel_cnt < high_th):
           stain_normalization_OD(img_rz,final_des_img) 
        counter = counter + 1
    return 0


def randomly_selected_samples_train_or_val(data_path, patch_h, patch_w, number_of_samples, img_saving_dir):
        
    for path2, subdirs2, files2 in os.walk(data_path):
        for name in subdirs2:
            img_dir = os.path.join(data_path,name)
            img_dir = img_dir+'/' 
            images = [x for x in sorted(os.listdir(img_dir)) if x[-4:] == '.jpg' or '.tif' or '.png']      
            total = len(images) 
            print('Total number of sampels: '+ str(total))
            rand_sample_ids = random.sample(range(0,total),number_of_samples)
            print('Total random sampels: '+ str(rand_sample_ids)) 
            
            if not os.path.isdir("%s/%s"%(img_saving_dir,name)):
                os.makedirs("%s/%s"%(img_saving_dir,name))
	    # create all necessary path for saving log files
            img_saving_dir_final = join_path(img_saving_dir,name)            
            print(img_saving_dir_final+'/') 
            
            counter = 0
            for x in range(number_of_samples):
                image_id = rand_sample_ids[x]
                print('Randomly selected idx: '+ str(image_id))             
                image_name = images[image_id]   
                img_name = image_name.split('.')[0]  
                patch_img = cv2.imread(img_dir + img_name+'.png', cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')                      
                f_img_name =str(img_name)+'.png' 
                final_des_img = os.path.join(img_saving_dir_final,f_img_name)
                cv2.imwrite(final_des_img,patch_img)
                counter = counter + 1
    return 0





def randomly_selected_images_mask_256x256(data_path, patch_h, patch_w, number_of_samples, img_saving_dir_train,img_saving_dir_val):
    
    images = [x for x in sorted(os.listdir(data_path)) if x[-9:] == '_mask.png']      
    total = len(images)            
   
    print('Total number of sampels: '+ str(total))   
    rand_sample_ids = random.sample(range(0,total),number_of_samples)
    print('Total random sampels: '+ str(rand_sample_ids))   
    
    counter = 0
    
    pdb.set_trace()
    
    for x in range(len(rand_sample_ids)):
        image_id = rand_sample_ids[x]
        print('Randomly selected idx: '+ str(image_id))             
        mask_name = images[image_id]   
        mask_name_woe = mask_name.split('.')[0] 
        image_name_woe = mask_name_woe.split('_m')[0]
        
        image = cv2.imread(data_path + image_name_woe+'.png', cv2.IMREAD_UNCHANGED).astype('float32')  
        mask_img = cv2.imread(os.path.join(data_path, mask_name), cv2.IMREAD_GRAYSCALE)
        
        counter = counter+1           
        f_img_name =str(image_name_woe)+'.png'
        
        if counter<=120000:
            final_des_img = os.path.join(img_saving_dir_train,f_img_name)        
            final_des_msk = os.path.join(img_saving_dir_train,mask_name)        
            cv2.imwrite(final_des_img,image)
            cv2.imwrite(final_des_msk,mask_img)
        else:
            final_des_img = os.path.join(img_saving_dir_val,f_img_name)        
            final_des_msk = os.path.join(img_saving_dir_val,mask_name)        
            cv2.imwrite(final_des_img,image)
            cv2.imwrite(final_des_msk,mask_img)


if __name__== "__main__":
    
  #create dataset and read training data...
  # MDC 2014 dataset...patche size for classification....
    
  patch_h = 256
  patch_w = 256
  
  data_path ="/research/rgs01/project_space/orrgrp/medulloblastoma/common/MZA/40X_HPF_4096/val/"
  img_saving_dir = "/research/rgs01/project_space/orrgrp/medulloblastoma/common/MZA/40X_HPF_256/train_val_final/val_all_images/"
  #create_train_val_database_WSG34_NMT20_SCD(data_path, patch_h, patch_w, img_saving_dir)
  
  data_path = "/research/rgs01/project_space/orrgrp/medulloblastoma/common/MZA/40X_HPF_256/train_val_final/val/"
  #img_saving_dir = "/research/rgs01/project_space/orrgrp/medulloblastoma/common/MZA/40X_HPF_256/train_val_final/train_rdm_samples/"
  img_saving_dir = "/research/rgs01/project_space/orrgrp/medulloblastoma/common/MZA/40X_HPF_256/train_val_final/val_rdm_samples/"
  number_of_samples = 5000
  #randomly_selected_samples_train_or_val(data_path, patch_h, patch_w, number_of_samples, img_saving_dir)
  #pdb.set_trace()
  
  data_path = "/research/rgs01/project_space/orrgrp/medulloblastoma/common/MZA/40X_HPF_256/train_val_final/train_rdm_samples/3/"
  img_saving_dir = "/research/rgs01/project_space/orrgrp/medulloblastoma/common/MZA/40X_HPF_256/train_val_final/train_rdm_samples_norm/3/"
  
  #data_path = "/research/rgs01/project_space/orrgrp/medulloblastoma/common/MZA/40X_HPF_256/train_val_final/test0/"
  #img_saving_dir = "/research/rgs01/project_space/orrgrp/medulloblastoma/common/MZA/40X_HPF_256/train_val_final/out/"
  #stain_norm_and_save_train_or_val_dir(data_path, patch_h, patch_w,img_saving_dir)


  data_path = "/research/rgs01/project_space/orrgrp/medulloblastoma/common/MZA/40X_HPF_256/train_val_final/val_rdm_samples/0/"
  img_saving_dir = "/research/rgs01/project_space/orrgrp/medulloblastoma/common/MZA/40X_HPF_256/train_val_final/val_rdm_samples_norm/0/"
  #stain_norm_and_save_train_or_val_dir(data_path, patch_h, patch_w,img_saving_dir)
  
  data_path = "/research/rgs01/project_space/orrgrp/medulloblastoma/common/MZA/40X_HPF_4096_selected/val/"
  img_saving_dir = "/research/rgs01/project_space/orrgrp/medulloblastoma/common/MZA/40X_Patches_256_selected/val/"
  create_training_patches_from_sub_sub_dir_blue_ratio_normalized(data_path, patch_h, patch_w, img_saving_dir)
  
  



























