# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:54:37 2019
@author: deeplens
"""
import openslide
#from scipy.misc import imsave, imresize
#import imageio
from openslide import open_slide # http://openslide.org/api/python/
import numpy as np
import os
import pdb
import json
import cv2
from matplotlib import pyplot as plt

#from scipy.misc import imsave
from os.path import join as join_path
abspath = os.path.dirname(os.path.abspath(__file__))

#kernel = np.ones((5,5), np.uint8) # rectangular kernel..
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))

save = True

valid_images = ['.svs','.tif','.jpg']


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

def extract_patches_from_image(full_img,patch_h,patch_w, img_name, patches_saving_dir):

    height,width, channel = full_img.shape    
    rows = (int) (height/patch_h)
    columns = (int) (width/patch_w)       
    k = 0
    pn = 0
    for r_s in range(rows):
        for c_s in range(columns):
            patch_img = full_img[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w,:]
            f_img_name =str(pn)+'.jpg'
            final_des_img = os.path.join(patches_saving_dir,f_img_name)            
            mx_val = patch_img.max()
            mn_val = patch_img.min()            
            print ('max_val : '+str(mx_val))
            print ('min_val : '+str(mn_val))          
            cv2.imwrite(final_des_img,patch_img)
            #cv2.imwrite(final_des_mask,patch_mask)
            pn+=1
        k +=1
        print ('Processing for: ' +str(k))
        
    return pn

def extract_image_seq_non_overlapped_patches_blue_ratio(HPF_img,patch_h,patch_w, img_name, imd_saving_dir):   
    
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
                #stain_normalization_OD(patch_img,final_des_img)
                cv2.imwrite(final_des_img,patch_img)

            pn+=1           
        k +=1
        print ('Processing for: ' +str(k))
        
    return pn

def prepare_steatosis_patches_with_blue_ratio_for_seg(HPF_img,patch_h,patch_w, img_name, imd_saving_dir):   
    
 
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
            
            #threshold = (max_val+avg_val)/2  
            threshold = avg_val#+std_val  

            binary_image = 1.0 * (m_blue_ratio_image > threshold)
            pred = binary_image
            pred_mask = cv2.dilate(pred,kernel,iterations = 1)            
            pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)    #  To fillup the internal pixels...
            pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)     # opening operation to remove the noise 
            pred_mask = cv2.erode(pred_mask,kernel,iterations = 1)    
            morph_pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
            mask = 255-(morph_pred_mask*255)                 
            
            white_pixel_cnt = cv2.countNonZero(morph_pred_mask)                       
            total_pixel = float(patch_h*patch_w)
            percent_wpxls = white_pixel_cnt/total_pixel
            
            print('Total white pixels:'+str(white_pixel_cnt))
            print('Percent of pixels ',percent_wpxls)
            
            f_txt_name =str(img_name)+'_'+str(pn)+'.txt'
            final_des_txt = os.path.join(imd_saving_dir,f_txt_name)
            #pdb.set_trace()
            file_name = open(final_des_txt,"w")
            file_name.write("Total pixels is: " + str(total_pixel))
            file_name.write("\n Total pixels for steatosis is: " + str(total_pixel-white_pixel_cnt))
            file_name.write("\n The percentage of pixel is: " + str(1-percent_wpxls))
            file_name.close()
            
            f_img_name =str(img_name)+'_'+str(pn)+'.png'
            final_des_img = os.path.join(imd_saving_dir,f_img_name)
            f_mask_name =str(img_name)+'_'+str(pn)+'_mask.png'
            final_des_mask = os.path.join(imd_saving_dir,f_mask_name)
            #if white_pixel_cnt > ((patch_h*patch_w) * 0.20):
            #    cv2.imwrite(final_des_img,patch_img)

            low_th = int((patch_h*patch_w)*0.35)
            high_th = int((patch_h*patch_w)*0.9)
            #if max_val>50 and (low_th<white_pixel_cnt and white_pixel_cnt < high_th):
                #stain_normalization_OD(patch_img,final_des_img)
            cv2.imwrite(final_des_img,patch_img)
            cv2.imwrite(final_des_mask,mask)
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
                        #num_patches = extract_image_seq_non_overlapped_patches_blue_ratio_normalized(input_image,patch_h,patch_w, img_name, final_img_saving_dir)                        
                        num_patches = extract_image_seq_non_overlapped_patches_blue_ratio_normalized(input_image,patch_h,patch_w, img_name, final_img_saving_dir)                        

                        print ('Processing done for: ' +str(i))

def create_training_patches_from_sub_dir_blue_ratio(data_path, patch_h, patch_w, img_saving_dir):
    
    for path, subdirs, files in os.walk(data_path):        
        for dir_name in subdirs:            
            sub_dir_path = os.path.join(path, dir_name+'/')           
            print(dir_name)    
            
            if not os.path.isdir("%s/%s"%(img_saving_dir,dir_name)):
                os.makedirs("%s/%s"%(img_saving_dir,dir_name))                
            final_img_saving_dir = join_path(img_saving_dir,dir_name+'/')
                 
            # Checkk the sampels and read images for sub-patching....
            images = [x for x in sorted(os.listdir(sub_dir_path)) if x[-4:] == '.jpg' or '.png' or'.tif']    
            
            for i, img_name in enumerate(images):               
                acc_name = img_name.split('.')[0]  
                img_ext = img_name.split('.')[1]
                input_image = cv2.imread(sub_dir_path + img_name, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')                                
                img_name = dir_name+'_'+acc_name
                num_patches = extract_image_seq_non_overlapped_patches_blue_ratio(input_image,patch_h,patch_w, img_name, final_img_saving_dir)
                
                print ('Processing done for: ' +str(i))

def create_seg_training_patches_from_sub_dir_blue_ratio(data_path, patch_h, patch_w, img_saving_dir):
    
    for path, subdirs, files in os.walk(data_path):        
        for dir_name in subdirs:            
            sub_dir_path = os.path.join(path, dir_name+'/')           
            print(dir_name)    
            
            if not os.path.isdir("%s/%s"%(img_saving_dir,dir_name)):
                os.makedirs("%s/%s"%(img_saving_dir,dir_name))                
            final_img_saving_dir = join_path(img_saving_dir,dir_name+'/')
                 
            # Checkk the sampels and read images for sub-patching....
            images = [x for x in sorted(os.listdir(sub_dir_path)) if x[-4:] == '.jpg' or '.png' or '.tif']    
            
            #pdb.set_trace()
            
            for i, img_name in enumerate(images):               
                acc_name = img_name.split('.')[0]  
                img_ext = img_name.split('.')[1]
                input_image = cv2.imread(sub_dir_path + img_name, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')                                
                img_name = dir_name+'_'+acc_name
                prepare_steatosis_patches_with_blue_ratio_for_seg(input_image,patch_h,patch_w, img_name, final_img_saving_dir)
                
                print ('Processing done for: ' +str(i))



def create_training_patches_from_sub_sub_dir_blue_ratio(data_path, patch_h, patch_w, img_saving_dir):
                  
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
                    images = [x for x in sorted(os.listdir(sub_sub_dir)) if x[-4:] == '.jpg']   
                    #pdb.set_trace()
                    for i, img_name in enumerate(images):               
                        acc_name = img_name.split('.')[0]  
                        #print(acc_name)
                        #img_ext = img_name.split('.')[1]
                        input_image = cv2.imread(sub_sub_dir + img_name, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')                                
                        #img_name = dir_name+'_'+dir_name_2+'_'+acc_name
                        img_name = dir_name+'_'+acc_name

                        #num_patches = extract_image_seq_non_overlapped_patches_blue_ratio_normalized(input_image,patch_h,patch_w, img_name, final_img_saving_dir)                        
                        num_patches = extract_image_seq_non_overlapped_patches_blue_ratio(input_image,patch_h,patch_w, img_name, final_img_saving_dir)                        

                        print ('Processing done for: ' +str(i))
                        
def extract_same_size_patches_from_svs_final(svs_img_dir, patches_saving_dir, wsi_mask_dir, patch_size):
            
    patch_dir_name = 'patches_'+str(patch_size[0])+'/'
        
    if not os.path.isdir("%s/%s"%(patches_saving_dir,patch_dir_name)):
        os.makedirs("%s/%s"%(patches_saving_dir,patch_dir_name))        
    
    patches_dir = join_path(patches_saving_dir+patch_dir_name)
    
    image_svs = [x for x in sorted(os.listdir(svs_img_dir)) if x[-4:] == '.svs' or '.tif']
  
    #for f in os.listdir(image_svs):
    for i, f in enumerate(image_svs):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        
        dir_name = os.path.splitext(f)[0]              
        print('Processing for :' + str(dir_name))     
        if not os.path.isdir("%s/%s"%(patches_dir,dir_name)):
            os.makedirs("%s/%s"%(patches_dir,dir_name))
        
        patches_sub_dir = join_path(patches_dir+dir_name+'/')
        
 
        # open scan
        svs_img_path = os.path.join(svs_img_dir,f)
        scan = openslide.OpenSlide(svs_img_path)
        
        # mask name .... 
        mask_name = dir_name+'_mask.tif'
        svs_mask_path = os.path.join(wsi_mask_dir,mask_name)
        scan_mask = openslide.OpenSlide(svs_mask_path)

        pdb.set_trace()  
        
        scan_dimensions = scan.dimensions        
        orig_w = scan_dimensions[0]
        orig_h = scan_dimensions[1]   
        
        print('The dimension of image: ('+str(orig_w)+','+str(orig_h)+')')
        
        no_patches_x_axis = orig_w/patch_size[0]
        no_patches_y_axis = orig_h/patch_size[1]                

        starting_row_columns = []
        img_saving_idx = 0        

        for y in range(0,orig_h,patch_size[1]):
            for x in range(0, orig_w,patch_size[0]):                
                # save only those HPF patches that satify the following condition...
                if x+patch_size[0] <= orig_w and y+patch_size[1] <= orig_h:
                    img = np.array(scan.read_region((x,y),0,(patch_size[0],patch_size[1])),dtype=np.uint8)[...,0:3]  
                    mask = np.array(scan_mask.read_region((x,y),0,(patch_size[0],patch_size[1])),dtype=np.uint8)[...,0]                   
                
                print("Processing for :"+str(dir_name)+'  coordinate : ('+str(x)+','+str(y)+')')
                
                mask = np.array(mask)
                white_pixel_cnt = cv2.countNonZero(mask)
                
                #if white_pixel_cnt > ((patch_size[0]*patch_size[1]) * 0.05): 
                if white_pixel_cnt > 0 :                

                    idx_sr_sc = str(img_saving_idx)+','+str(x)+','+str(y)                
                    starting_row_columns.append(idx_sr_sc)
                    print("Processing:"+str(img_saving_idx))                
                    ac_img_name =str(dir_name)+'_'+str(img_saving_idx)+'.jpg'
                    mask_name = str(dir_name)+'_'+str(img_saving_idx)+'_mask.jpg'
                    
                    final_img_des = os.path.join(patches_sub_dir,ac_img_name)
                    final_mask_des = os.path.join(patches_sub_dir,mask_name)

                    #cv2.imwrite(final_img_des,img)
                    #imsave(final_img_des,img)  
                    cv2.imwrite(final_img_des,img) 
                    #imsave(final_mask_des,mask)
                    cv2.imwrite(final_mask_des,mask) 

                img_saving_idx +=1
                
        scan.close
        scan_mask.close
    
        svs_log = {}
        svs_log["ID"] = dir_name
        svs_log["height"] = orig_h
        svs_log["width"] = orig_w
        svs_log["patch_width"] = patch_size[0]
        svs_log["patch_height"] = patch_size[1]
        svs_log["no_patches_x_axis"] = no_patches_x_axis
        svs_log["no_patches_y_axis"] = no_patches_y_axis
        svs_log["number_HPFs_patches"] = img_saving_idx
        svs_log["starting_rows_columns"] = starting_row_columns
         
        # make experimental log saving path...
        json_file = os.path.join(patches_sub_dir,'image_patching_log.json')
        with open(json_file, 'w') as file_path:
            json.dump(svs_log, file_path, indent=4, sort_keys=True)
        
        
    return patches_sub_dir

def generate_steatosis_final_mask(mask, start_x,start_y,max_conf_values,class_indexes, patch_h, patch_w, HPF_height, HPF_width):

    final_mask = np.zeros((HPF_height,HPF_width),dtype=int)
    for k in range(len(start_x)):        
        row = int(start_x[k]*patch_h)
        column = int(start_y[k]*patch_w)
        index_value = int(class_indexes[k])
        
        if index_value <= 3:  #Bronchiacell(Green)
            indv_mask = mask [row:row+patch_h, column: column+patch_w]
            final_mask [row:row+patch_h, column: column+patch_w ] = indv_mask            
        else:
            indv_mask = mask [row:row+patch_h, column: column+patch_w]
            indv_mask = (indv_mask/255)*0              
            final_mask [row:row+patch_h, column: column+patch_w ] = indv_mask 
    
    return final_mask

def extract_same_size_patches_from_wsi_final(svs_img_dir, patches_saving_dir, patch_size):
    
           
#    patch_dir_name = 'patches_'+str(patch_size[0])+'/ 
#    if not os.path.isdir("%s/%s"%(patches_saving_dir,patch_dir_name)):
#        os.makedirs("%s/%s"%(patches_saving_dir,patch_dir_name))        
#    patches_dir = join_path(patches_saving_dir+patch_dir_name)
#    
    patches_dir = patches_saving_dir
    
    image_svs = [x for x in sorted(os.listdir(svs_img_dir)) if x[-4:] == '.tif']
    
    #pdb.set_trace()    
    #for f in os.listdir(image_svs):
    for i, f in enumerate(image_svs):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        
        dir_name = os.path.splitext(f)[0]              
        print(svs_img_dir.split('/')[0])        
        if not os.path.isdir("%s/%s"%(patches_dir,dir_name)):
            os.makedirs("%s/%s"%(patches_dir,dir_name))
        
        patches_sub_dir = join_path(patches_dir,dir_name+'/')
        # open scan
        svs_img_path = os.path.join(svs_img_dir,f)
        scan = openslide.OpenSlide(svs_img_path)
        
        scan_dimensions = scan.dimensions        
        orig_w = scan_dimensions[0]
        orig_h = scan_dimensions[1]           
        no_patches_x_axis = orig_w/patch_size[0]
        no_patches_y_axis = orig_h/patch_size[1]                

        starting_row_columns = []
        img_saving_idx = 0        

        for y in range(0,orig_h,patch_size[1]):
            for x in range(0, orig_w,patch_size[0]):                
                # save only those HPF patches that satify the following condition...
                if x+patch_size[0] <= orig_w and y+patch_size[1] <= orig_h:
                    img = np.array(scan.read_region((x,y),0,(patch_size[0],patch_size[1])),dtype=np.uint8)[...,0:3]                   
                
                idx_sr_sc = str(img_saving_idx)+','+str(x)+','+str(y)                
                starting_row_columns.append(idx_sr_sc)
                print("Processing:"+str(img_saving_idx))                
                ac_img_name =dir_name+'_'+str(img_saving_idx)+'.jpg'
                final_img_des = os.path.join(patches_sub_dir,ac_img_name)
                cv2.imwrite(final_img_des,img)
                #imsave(final_img_des,img)                
                img_saving_idx +=1
                
        scan.close 
    
        #pdb.set_trace()
         
        svs_log = {}
        svs_log["ID"] = dir_name
        svs_log["height"] = orig_h
        svs_log["width"] = orig_w
        svs_log["patch_width"] = patch_size[0]
        svs_log["patch_height"] = patch_size[1]
        svs_log["no_patches_x_axis"] = no_patches_x_axis
        svs_log["no_patches_y_axis"] = no_patches_y_axis
        svs_log["number_HPFs_patches"] = img_saving_idx
        svs_log["starting_rows_columns"] = starting_row_columns
         
        # make experimental log saving path...
        json_file = os.path.join(patches_sub_dir,'image_patching_log.json')
        with open(json_file, 'w') as file_path:
            json.dump(svs_log, file_path, indent=4, sort_keys=True)
        
        
    return patches_sub_dir

    
def extract_same_size_patches_from_normal_wsi(svs_img_dir, patches_saving_dir, patch_size):
            
#    patch_dir_name = 'patches_normal'+str(patch_size[0])+'/'
#        
#    if not os.path.isdir("%s/%s"%(patches_saving_dir,patch_dir_name)):
#        os.makedirs("%s/%s"%(patches_saving_dir,patch_dir_name))        
#    
#    patches_dir = join_path(patches_saving_dir+patch_dir_name)
    
    patches_dir = patches_saving_dir
    
    image_svs = [x for x in sorted(os.listdir(svs_img_dir)) if x[-4:] == '.svs' or '.tif']
  
    #for f in os.listdir(image_svs):
    for i, f in enumerate(image_svs):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        
        dir_name = os.path.splitext(f)[0]              
        print(svs_img_dir.split('/')[0])        
        if not os.path.isdir("%s/%s"%(patches_dir,dir_name)):
            os.makedirs("%s/%s"%(patches_dir,dir_name))
        
        patches_sub_dir = join_path(patches_dir+dir_name+'/')
        
 
        # open scan
        svs_img_path = os.path.join(svs_img_dir,f)
        scan = openslide.OpenSlide(svs_img_path)
        
        # mask name .... 
        #mask_name = dir_name+'_mask.tif'
        #svs_mask_path = os.path.join(svs_img_dir,mask_name)
        #scan_mask = openslide.OpenSlide(svs_mask_path)

        #pdb.set_trace()  
        
        scan_dimensions = scan.dimensions        
        orig_w = scan_dimensions[0]
        orig_h = scan_dimensions[1]   
        
        print('The dimension of image: ('+str(orig_w)+','+str(orig_h)+')')
        
        no_patches_x_axis = orig_w/patch_size[0]
        no_patches_y_axis = orig_h/patch_size[1]                

        starting_row_columns = []
        img_saving_idx = 0        

        for y in range(0,orig_h,patch_size[1]):
            for x in range(0, orig_w,patch_size[0]):                
                # save only those HPF patches that satify the following condition...
                if x+patch_size[0] <= orig_w and y+patch_size[1] <= orig_h:
                    img = np.array(scan.read_region((x,y),0,(patch_size[0],patch_size[1])),dtype=np.uint8)[...,0:3]  
                    #mask = np.array(scan_mask.read_region((x,y),0,(patch_size[0],patch_size[1])),dtype=np.uint8)[...,0]  
                    
                    patch_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    lower_red = np.array([20, 20, 20])
                    upper_red = np.array([200, 200, 200])
                    mask_patch = cv2.inRange(patch_hsv, lower_red, upper_red)
                    white_pixel_cnt = cv2.countNonZero(mask_patch)
                    
                
                    print("Processing for :"+str(dir_name)+'  coordinate : ('+str(x)+','+str(y)+')')
                
                #mask = np.array(mask)
                #white_pixel_cnt = cv2.countNonZero(mask)
                
                    if white_pixel_cnt > ((patch_size[0]*patch_size[1]) * 0.05): 
    
                        idx_sr_sc = str(img_saving_idx)+','+str(x)+','+str(y)                
                        starting_row_columns.append(idx_sr_sc)
                        print("Processing:"+str(img_saving_idx))                
                        ac_img_name =str(dir_name)+'_'+str(img_saving_idx)+'.jpg'
                        #mask_name = str(dir_name)+'_'+str(img_saving_idx)+'_mask.jpg'
                        
                        final_img_des = os.path.join(patches_sub_dir,ac_img_name)
                        #final_mask_des = os.path.join(patches_sub_dir,mask_name)
    
                        cv2.imwrite(final_img_des,img)
                        #imsave(final_img_des,img)     
                    #imsave(final_mask_des,mask)
                    
                img_saving_idx +=1
                
        scan.close
        #scan_mask.close
    
        svs_log = {}
        svs_log["ID"] = dir_name
        svs_log["height"] = orig_h
        svs_log["width"] = orig_w
        svs_log["patch_width"] = patch_size[0]
        svs_log["patch_height"] = patch_size[1]
        svs_log["no_patches_x_axis"] = no_patches_x_axis
        svs_log["no_patches_y_axis"] = no_patches_y_axis
        svs_log["number_HPFs_patches"] = img_saving_idx
        svs_log["starting_rows_columns"] = starting_row_columns
         
        # make experimental log saving path...
        json_file = os.path.join(patches_sub_dir,str(dir_name)+'_patching_log.json')
        
        
        with open(json_file, 'w') as file_path:
            json.dump(svs_log, file_path, indent=4, sort_keys=True)
        
        
    return patches_sub_dir



def extract_same_size_patches_from_svs_v2(svs_img_dir, svs_mask_dir, patches_saving_dir, patch_size):
            
#    patch_dir_name = 'patches_'+str(patch_size[0])+'/'
#        
#    if not os.path.isdir("%s/%s"%(patches_saving_dir,patch_dir_name)):
#        os.makedirs("%s/%s"%(patches_saving_dir,patch_dir_name))        
#    
#    patches_dir = join_path(patches_saving_dir+patch_dir_name)
    patches_dir = patches_saving_dir
    image_svs = [x for x in sorted(os.listdir(svs_img_dir)) if x[-4:] == '.svs' or '.tif']
    
    #slide_paths = glob.glob(osp.join(svs_img_dir, '*.tif'))
    #slide_paths.sort()
    #mask_paths = glob.glob(osp.join(svs_mask_dir, '*.tif'))
    #mask_paths.sort()
    
    #pdb.set_trace()
    #for f in os.listdir(image_svs):
    for i, f in enumerate(image_svs):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        
        dir_name = os.path.splitext(f)[0]              
        print('Processing for : '+dir_name)     
        
        if not os.path.isdir("%s/%s"%(patches_dir,dir_name)):
            os.makedirs("%s/%s"%(patches_dir,dir_name))
        
        patches_sub_dir = join_path(patches_dir+dir_name+'/')
        
 
        # open scan
        svs_img_path = os.path.join(svs_img_dir,f)
        scan = openslide.OpenSlide(svs_img_path)
        
        # mask name .... 
        mask_name = dir_name+'_mask.tif'
        svs_mask_path = os.path.join(svs_mask_dir,mask_name)
        scan_mask = openslide.OpenSlide(svs_mask_path)

        #pdb.set_trace()  
        
        scan_dimensions = scan.dimensions        
        orig_w = scan_dimensions[0]
        orig_h = scan_dimensions[1]   
        
        print('The dimension of image: ('+str(orig_w)+','+str(orig_h)+')')
        
        no_patches_x_axis = orig_w/patch_size[0]
        no_patches_y_axis = orig_h/patch_size[1]                

        starting_row_columns = []
        patch_idx = 0        
        patch_idx_tumor = 0
        
        for y in range(0,orig_h,patch_size[1]):
            for x in range(0, orig_w,patch_size[0]):                
                # save only those HPF patches that satify the following condition...
                if x+patch_size[0] <= orig_w and y+patch_size[1] <= orig_h:
                    img = np.array(scan.read_region((x,y),0,(patch_size[0],patch_size[1])),dtype=np.uint8)[...,0:3]  
                    mask = np.array(scan_mask.read_region((x,y),0,(patch_size[0],patch_size[1])),dtype=np.uint8)[...,0]                   
                
                print("Processing for :"+str(dir_name)+'  coordinate : ('+str(x)+','+str(y)+')')
                
                mask = np.array(mask)
                white_pixel_cnt = cv2.countNonZero(mask)
                
                #if white_pixel_cnt > ((patch_size[0]*patch_size[1]) * 0.05): 
                
                if white_pixel_cnt > 0 :                

                    idx_sr_sc = str(patch_idx)+','+str(x)+','+str(y)                
                    starting_row_columns.append(idx_sr_sc)
                    print("Processing:"+str(patch_idx))                
                    ac_img_name =str(dir_name)+'_'+str(patch_idx)+'.jpg'
                    mask_name = str(dir_name)+'_'+str(patch_idx)+'_mask.jpg'
                    
                    final_img_des = os.path.join(patches_sub_dir,ac_img_name)
                    final_mask_des = os.path.join(patches_sub_dir,mask_name)

                    cv2.imwrite(final_img_des,img)
                    #imsave(final_img_des,img)     
                    #imsave(final_mask_des,mask)
                    cv2.imwrite(final_mask_des,mask)
                    patch_idx_tumor = patch_idx_tumor+1
                    
                patch_idx +=1
                
        scan.close
        scan_mask.close
    
        svs_log = {}
        svs_log["ID"] = dir_name
        svs_log["height"] = orig_h
        svs_log["width"] = orig_w
        svs_log["patch_width"] = patch_size[0]
        svs_log["patch_height"] = patch_size[1]
        svs_log["no_patches_x_axis"] = no_patches_x_axis
        svs_log["no_patches_y_axis"] = no_patches_y_axis
        svs_log["number_HPFs_patches"] = patch_idx_tumor
        svs_log["starting_rows_columns"] = starting_row_columns
         
        # make experimental log saving path...
        json_file = os.path.join(patches_sub_dir,str(dir_name)+'_patching_log.json')
        with open(json_file, 'w') as file_path:
            json.dump(svs_log, file_path, indent=4, sort_keys=True)
        
        
    return patches_dir
    
    
    
def extract_all_patches_from_wsi(svs_img_dir, patches_saving_dir, patch_size):
            
    patch_dir_name = 'patches_'+str(patch_size[0])+'/'
        
    if not os.path.isdir("%s/%s"%(patches_saving_dir,patch_dir_name)):
        os.makedirs("%s/%s"%(patches_saving_dir,patch_dir_name))        
    
    patches_dir = join_path(patches_saving_dir+patch_dir_name)
    
    image_svs = [x for x in sorted(os.listdir(svs_img_dir)) if x[-4:] == '.svs' or '.tif']
        
    #for f in os.listdir(image_svs):
    for i, f in enumerate(image_svs):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        
        dir_name = os.path.splitext(f)[0]              
        print(svs_img_dir.split('/')[0])        
        if not os.path.isdir("%s/%s"%(patches_dir,dir_name)):
            os.makedirs("%s/%s"%(patches_dir,dir_name))
        
        patches_sub_dir = join_path(patches_dir+dir_name+'/')
        # open scan
        svs_img_path = os.path.join(svs_img_dir,f)
        scan = openslide.OpenSlide(svs_img_path)
        
        scan_dimensions = scan.dimensions
        
        orig_w = scan_dimensions[0]
        orig_h = scan_dimensions[1]
        #orig_w = np.int(scan.properties.get('aperio.OriginalWidth'))
        #orig_h = np.int(scan.properties.get('aperio.OriginalHeight'))               
        no_patches_x_axis = orig_w/patch_size[0]
        no_patches_y_axis = orig_h/patch_size[1]                
        # create an array to store our image
        #img_np = np.zeros((orig_w,orig_h,3),dtype=np.uint8)        
        #pdb.set_trace()
        starting_row_columns = []

        img_saving_idx = 0
        
        for r in range(0,orig_h,patch_size[1]):
            for c in range(0, orig_w,patch_size[0]):
                
                if c+patch_size[1] > orig_w and r+patch_size[0]<= orig_h:
                    p = orig_w-c
                    img = np.array(scan.read_region((c,r),0,(p,patch_size[1])),dtype=np.uint8)[...,0:3]
                elif c+patch_size[1] <= orig_w and r+patch_size[0] > orig_h:
                    p = orig_h-r
                    img = np.array(scan.read_region((c,r),0,(patch_size[0],p)),dtype=np.uint8)[...,0:3]
                elif  c+patch_size[1] > orig_w and r+patch_size[0] > orig_h:
                    p = orig_h-c
                    pp = orig_w-r
                    img = np.array(scan.read_region((c,r),0,(p,pp)),dtype=np.uint8)[...,0:3]
                else:    
                    img = np.array(scan.read_region((c,r),0,(patch_size[0],patch_size[1])),dtype=np.uint8)[...,0:3]
                 
                idx_sr_sc = str(img_saving_idx)+','+str(c)+','+str(r)                
                starting_row_columns.append(idx_sr_sc)
                print("Processing:"+str(img_saving_idx))                
                ac_img_name =str(img_saving_idx)+'.jpg'
                final_img_des = os.path.join(patches_sub_dir,ac_img_name)
                cv2.imwrite(final_img_des,img)
                #imsave(final_img_des,img)   
                img_saving_idx +=1
              
                
                
                
        scan.close
    
        svs_log = {}
        svs_log["ID"] = dir_name
        svs_log["height"] = orig_h
        svs_log["width"] = orig_w
        svs_log["patch_width"] = patch_size[0]
        svs_log["patch_height"] = patch_size[1]
        svs_log["no_patches_x_axis"] = no_patches_x_axis
        svs_log["no_patches_y_axis"] = no_patches_y_axis
        svs_log["number_HPFs_patches"] = img_saving_idx
        svs_log["starting_rows_columns"] = starting_row_columns
         
        # make experimental log saving path...
        json_file = os.path.join(patches_sub_dir,'image_patching_log.json')
        with open(json_file, 'w') as file_path:
            json.dump(svs_log, file_path, indent=4, sort_keys=True)
        
        
    return patches_sub_dir
    

def patches_to_image(patches_dir):
    
    image_id = []
    image_h = []
    image_w = []
    patch_h = []
    patch_w = []
    
   
    json_files = [x for x in sorted(os.listdir(patches_dir)) if x[-5:] == '.json']           
    image_files = [x for x in sorted(os.listdir(patches_dir)) if x[-4:] == '.jpg']
    
    
    names_wo_ext = []
    
    #pdb.set_trace()
    
    for idx in range(len(image_files)):
        name = image_files[idx]
        name_wo_ext = name.split('.')[0]
        names_wo_ext.append(name_wo_ext)
    
   #extension = '.jpg'    
    patches_name_wo_ext = np.array(names_wo_ext)       
    patches_name_wo_ext.sort()
    #patches_name_wo_ext = patches_name_wo_ext.tolist()
    
    json_path = join_path(patches_dir,json_files[0])
    
    if len(json_files)<= 0 :
         print("The json file is not available")
    else:
        with open(json_path, "r") as f:
            image_logs = json.load(f)          

        image_id = image_logs["ID"]
        image_h = int(image_logs["height"])
        image_w = int(image_logs["width"])
        patch_w =  int(image_logs["patch_width"])
        patch_h =  int(image_logs["patch_height"])        
        num_rows =  int(image_logs["no_patches_x_axis"])
        num_columns =  int(image_logs["no_patches_y_axis"])
            
    img_from_patches = np.zeros((image_w,image_h,3),dtype=np.uint8)

    
    #pdb.set_trace()     
    patch_idx = 0        
    for r in range(0,num_rows):
        for c in range(0, num_columns):
            name = str(patches_name_wo_ext[patch_idx])+'.jpg'
            patch = cv2.imread(patches_dir + name, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
            #img_from_patches[r,c]=patch  
            print(patch.mean())
            print(name)
            print(patch.shape)
            
            img_from_patches[r*patch_h: r*patch_h+patch_h,c*patch_w:c*patch_w+patch_w,:] = patch
            
            print("Merging patch no. :"+str(patch_idx))               
            patch_idx +=1
    
    #pdb.set_trace()
    
    #print(img_from_patches.shape)
    
    resized_img = cv2.resize(img_from_patches, (5120,5120), interpolation=cv2.INTER_LANCZOS4)
    img_name =str(image_id)+'_merge.jpg'
    final_img_des = os.path.join(patches_dir,img_name)
    #imsave(final_img_des,resized_img)
    cv2.imwrite(final_img_des,resized_img)
    
    
def patch2subpatches_driver(patches_source, patches_saving_dir,patch_size):
    
    #pdb.set_trace()
    
    dir_name = "patches_"+str(patch_size[0])
    if not os.path.isdir("%s/%s"%(patches_saving_dir,dir_name)):
        os.makedirs("%s/%s"%(patches_saving_dir,dir_name))
    
    patch_dir_name = 'patches_'+str(patch_size[0])+'/'    
    patches_dir = join_path(patches_saving_dir,patch_dir_name)    
    image_dirs = [x for x in sorted(os.listdir(patches_source)) if x[-4:] == '.jpg']
    #for f in os.listdir(image_svs):
    for i, f in enumerate(image_dirs):
        
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        
        dir_name = os.path.splitext(f)[0]               
        img_name = f
        print(patches_source.split('/')[0])        
        if not os.path.isdir("%s/%s"%(patches_dir,dir_name)):
            os.makedirs("%s/%s"%(patches_dir,dir_name))
        
        patches_sub_dir = join_path(patches_dir+dir_name+'/')
        # open scan
        img_path = os.path.join(patches_source,f)
        
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
        
        orig_w,orig_h,channels = img.shape
        no_patches_x_axis = orig_w/patch_size[0]
        no_patches_y_axis = orig_h/patch_size[1]
                
        svs_log = {}
        svs_log["ID"] = dir_name
        svs_log["height"] = orig_h
        svs_log["width"] = orig_w
        svs_log["patch_width"] = patch_size[0]
        svs_log["patch_height"] = patch_size[1]
        svs_log["no_patches_x_axis"] = no_patches_x_axis
        svs_log["no_patches_y_axis"] = no_patches_y_axis
        
        # make experimental log saving path...
        json_file = os.path.join(patches_sub_dir,'image_log.json')
        with open(json_file, 'w') as file_path:
            json.dump(svs_log, file_path, indent=4, sort_keys=True)
        
        patches_number = extract_patches_from_image(img,patch_size[0],patch_size[1], img_name, patches_sub_dir)
   
        print(str(patches_number))