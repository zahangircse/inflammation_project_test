# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:26:08 2019

@author: deeplens
"""
import pdb
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pandas as pd
import json
import glob
from scipy.io import loadmat

import cv2

import scipy.ndimage as ndimage
from os.path import join as join_path
from PIL import Image


kernel = np.ones((13,13), np.uint8) 

abspath = os.path.dirname(os.path.abspath(__file__))

height = 1040
width = 1392

patch_h = 32
patch_w = 32

def saving_image_in_subdirs(image_path,image_saving_dir):            
   
    all_images = [x for x in sorted(os.listdir(image_path)) if x[-4:] == '.tif']  
    
    pdb.set_trace()
    
    for i, name in enumerate(all_images):  
            
        im = cv2.imread(image_path + name, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
        f_img_name_wo_extension = name.split('.')[0]

        image_id = f_img_name_wo_extension.split('-')[1] 
        
        height, width = im.shape        
        mask_initial = np.zeros((height, width), dtype='float32')
        
        if not os.path.isdir("%s/%s"%(image_saving_dir,f_img_name_wo_extension)):
            os.makedirs("%s/%s"%(image_saving_dir,f_img_name_wo_extension))
                    
        final_des_image = os.path.join(image_saving_dir,f_img_name_wo_extension)
        cv2.imwrite(final_des_image,im)            

        print ('Processing done for: '+f_img_name_wo_extension+'number: '+str(i))
           
    return 0

def extract_image_seq_non_overlapped_patches(full_img,full_mask,patch_h,patch_w, img_name, imd_saving_dir):
    
    #pdb.set_trace()   
    #print (full_img.shape)
    
    height,width = full_mask.shape
     
    rows = (int) (height/patch_h)
    columns = (int) (width/patch_w)

    k = 0
    pn = 0
    for r_s in range(rows):
        for c_s in range(columns):
#            if channel == 3:
#                patch_img = full_img[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w,:]
#                patch_mask = full_mask[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w,:]
#            else:
            patch_img = full_img[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w]
            patch_mask = full_mask[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w]
                
            f_img_name =str(img_name)+'_'+str(pn)+'.tif'
            f_mask_name =str(img_name)+'_'+str(pn)+'_mask'+'.tif'           
            final_des_img = os.path.join(imd_saving_dir,f_img_name)
            final_des_mask = os.path.join(imd_saving_dir,f_mask_name)
            
            mx_val = patch_mask.max()
            mn_val = patch_mask.min()
            
            print ('max_val : '+str(mx_val))
            print ('min_val : '+str(mn_val))
            
            #cv2.imwrite(final_des_img,patch_img)
            #cv2.imwrite(final_des_mask,patch_mask)
            

            if mx_val > 0:
                
                #final_des = join_path(imd_saving_dir,'1/')
                cv2.imwrite(final_des_img,patch_img)
                #Image.fromarray(patch_img).save(final_des_img)
                cv2.imwrite(final_des_mask,patch_mask)
                #Image.fromarray(patch_mask).save(final_des_mask)

            pn+=1
            
        k +=1
        print ('Processing for: ' +str(k))
    
    return pn
                           
                
def read_images_and_masks(data_path, image_h, image_w):
    
    
    train_data_path = os.path.join(data_path)
    images = glob.glob(train_data_path + "/*mask.tif")
    total = np.round(len(images)) 

    acc_imgs = np.ndarray((total, image_h, image_w,3), dtype=np.uint8)
    imgs = np.zeros((total, image_h, image_w), dtype=np.uint8)
    imgs_mask = np.zeros((total, image_h, image_w), dtype=np.uint8)
    
    i = 0
    print('Creating training images...')
    #img_patients = np.ndarray((total,), dtype=np.uint8)
    for image_name in images:

        image_mask_name = image_name.split('/')[-1]      
        img_first = image_mask_name.split('.')[0]
        img_second = img_first.split('_mask')[0]      
        image_name =img_second+'.tif'
                     
        acc_img = cv2.imread(os.path.join(train_data_path, image_name)) 
        acc_img_r = cv2.resize(acc_img, dsize=(image_h, image_w), interpolation=cv2.INTER_NEAREST)
        img = cv2.imread(os.path.join(train_data_path, image_name),cv2.IMREAD_GRAYSCALE)
        img_r = cv2.resize(img, dsize=(image_h, image_w), interpolation=cv2.INTER_NEAREST)
        mask_im = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
        mask_im_r = cv2.resize(mask_im, dsize=(image_h, image_w), interpolation=cv2.INTER_NEAREST)
        
        acc_imgs[i] = acc_img_r
        imgs[i] = img_r
        imgs_mask[i] = mask_im_r

        i += 1
        print ('Done',i)
    
    return acc_imgs,imgs,imgs_mask

def saving_patches(patches,image_name,images_saving_dir):    
    #patch saving...
         
    if len(patches.shape) > 2:
        num_patches,w,h = patches.shape        
        for k in range(num_patches):
            patch = patches[k,:,:]
            f_patch_name = str(image_name)+'_patch_'+str(k)+'.tif'          
            final_des_patch = os.path.join(images_saving_dir,f_patch_name)
            plt.imsave(final_des_patch, patch)            
    else:        
        f_patch_name = str(image_name)+'_patch'+'.tif'          
        final_des_patch = os.path.join(images_saving_dir,f_patch_name)
        plt.imsave(final_des_patch, patches)

def check_xy(x,y):
    if x< int(patch_w/2):
        x = width/2
    if x> int(width-patch_w/2):
        x= int(width-patch_w/2)    
    if y< int(patch_h/2):
        y = patch_h/2
    if y> int(height-patch_h/2):
        x= int(height-patch_h/2)
    
    return x,y

    
def mask_from_mat_CVPR_mitosis_detection_2019(mat_file, image_path, images_saving_dir): 
    
    directory_name = image_path.split('/')[-2]    
    #pdb.set_trace()    
    x = loadmat(mat_file)    
    np_array = np.array(x['result'][:])      
    print('Array length : '+str(len(np_array)))
    unique_np_array = np.unique(np_array[:,0])
    print('Unique value length : '+str(len(unique_np_array)))    

    print (x['result'].shape)
    samples,shape = x['result'].shape 
    
    for i in range(len(unique_np_array)):
        
        unique_idx = int(unique_np_array[i])        
        index_sets = np.array(np.argwhere(np_array[:,0] ==int(unique_idx)))        
        
        idx_4_img = index_sets[0] 
        image_id_xy = np_array[idx_4_img]           
        image_id = image_id_xy[0,0]         
        image_id = int(image_id)            
        if image_id < 10:
            image_id_final = '0000'+str(image_id)
        elif(image_id>=10 and image_id<100):
            image_id_final = '000'+str(image_id)
        elif(image_id>=100 and image_id<1000):
            image_id_final = '00'+str(image_id)        
                
        image_name = 'exp1_'+str(directory_name)+'-'+image_id_final  
        image_name_w_ext = image_name+'.tif'
        image_final_path = os.path.join(image_path,image_name_w_ext)        
    
        im = plt.imread(image_final_path)
        #view image
        #plt.imshow(im)
        #plt.show()
        image_mask = np.zeros((height, width), dtype='float32')                            
        idx_set_len = len(index_sets)        
        #pdb.set_trace()        
        if idx_set_len>1:
            
            patches = np.zeros((idx_set_len,patch_h,patch_w), dtype='float32')
            patch_idx = 0
            for k in range(idx_set_len):            
                idx = index_sets[k]            
                t_xy = np_array[idx]
                print (t_xy)                   
                t = t_xy[0,0]
                x = t_xy[0,1]
                y = t_xy[0,2]                
                x,y = check_xy(x,y)
                # extract patches...
                patches[patch_idx,:,:] =  cv2.resize((im[int(y-patch_h/2):int(y+patch_h/2),int(x-patch_w/2):int(x+patch_w/2)]),(patch_h,patch_w),interpolation=cv2.INTER_CUBIC)
                patch_idx =   patch_idx+1                              
                 # Generate mask for entire image...
                image_mask[int(y),int(x)] = 1 
            
            image_mask = 255.0*(image_mask[:,:]> 0)
            image_mask = cv2.dilate(image_mask,kernel,iterations = 1)
            image_mask = ndimage.gaussian_filter(image_mask, sigma=(1,1),order = 0)     
            image_mask = 255.0*(image_mask[:,:]> 0.3)
        else:
            idx = index_sets[0]            
            t_xy = np_array[idx]
            print (t_xy)  
            patches = np.zeros((patch_h,patch_w),dtype='float32')
         
            t = int(t_xy[0,0])
            x = int(t_xy[0,1])
            y = int(t_xy[0,2])
            
            x,y = check_xy(x,y)
            # extract patches...
            patches[:,:] = cv2.resize((im[int(y-patch_h/2):int(y+patch_h/2),int(x-patch_w/2):int(x+patch_w/2)]),(patch_h,patch_w),interpolation=cv2.INTER_CUBIC)
            
            # Generate mask for entire image...
            image_mask[int(y),int(x)] = 1 
            
            image_mask = 255.0*(image_mask[:,:]> 0)
            image_mask = cv2.dilate(image_mask,kernel,iterations = 1)
            image_mask = ndimage.gaussian_filter(image_mask, sigma=(1,1),order = 0)     
            image_mask = 255.0*(image_mask[:,:]> 0.3)
            
        mx_val = image_mask.max()
        mn_val = image_mask.min()
        print ('max_val : '+str(mx_val))
        print ('min_val : '+str(mn_val))
        
        #pdb.set_trace()
        
        f_img_name =str(image_name)+'.tif'
        f_mask_name =str(image_name)+'_mask'+'.tif' 
        f_patch_name_wo_ext = str(image_name)+'_patch'         
        final_des_img = os.path.join(images_saving_dir,f_img_name)
        final_des_mask = os.path.join(images_saving_dir,f_mask_name)                    
        cv2.imwrite(final_des_img,im)
        cv2.imwrite(final_des_mask,image_mask)
        
        # save patches...
        
        saving_patches(patches,image_name,images_saving_dir)


def create_dataset_seq_patches_driver(image_path,patch_h,patch_w,image_saving_path):
        

    all_images = glob.glob(image_path + "/*mask.tif")
    
    
    i = 0
    for image_name in all_images:

        image_mask_name = image_name.split('/')[-1]      
        img_first = image_mask_name.split('.')[0]
        img_second = img_first.split('_mask')[0] 
        image_name_wo_ext = img_second
        image_name =img_second+'.tif'
                     
        #im = cv2.imread(image_path + image_name, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')        
        im = plt.imread(image_path + image_name)        
        #mask_im = cv2.imread(os.path.join(image_path, image_mask_name), cv2.IMREAD_GRAYSCALE)        
        mask_im = plt.imread(image_path+image_mask_name)  
        #pdb.set_trace()
        mask_im = 255*(mask_im[:,:]>0)
        
        #pdb.set_trace()
        num_patches = extract_image_seq_non_overlapped_patches (im, mask_im, patch_h, patch_w, image_name_wo_ext, image_saving_path)
        
        #extract_image_seq_non_overlapped_patches(full_img,full_mask,patch_h,patch_w, img_name, imd_saving_dir)
        print ('Processing done for: ' +str(i))
        i = i+1
 
    return 0
    
if __name__== "__main__":
        
    mat_file_path = os.path.join(abspath,'annotation/MitosisAnnotations/090303_F0003.mat')
    image_path = os.path.join(abspath,'Images/images_masks/validation_images_masks/Group_4/F0015_images_masks/')    
    image_saving_path = os.path.join(abspath,'Images/images_masks/validation_images_masks/Group_4/F0015_patches_masks/') 
 
    # read image from original samples...  
    #mask_from_mat_CVPR_mitosis_detection_2019(mat_file_path,image_path,patches_saving_dir)   
    
    # Extract patches from images and mask.....
    
    
    #image_path = os.path.join(abspath,'Images/images_masks/training_images_masks/Group_1/F1_2_images_masks/')    
    
    patch_h = 128
    patch_w = 128
    
    #image_saving_path = os.path.join(abspath,'Images/images_masks/training_images_masks/Group_1/F1_2_patches_masks/') 
    
    create_dataset_seq_patches_driver(image_path,patch_h,patch_w,image_saving_path)
    
    #saving_image_in_subdirs(image_path,patches_saving_dir)
    
