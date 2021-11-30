#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 20:57:07 2018
@author: zahangir
"""
import numpy as np
import os
import glob
#from keras.preprocessing.image import ImageDataGenerator
import cv2
from os.path import join as join_path
import pdb
from collections import defaultdict
from skimage.transform import resize
#import shutil
import scipy.ndimage as ndimage

kernel = np.ones((7,7), np.uint8) 

allowed_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp','*.mat','*.tif']

#abspath = os.path.dirname(os.path.abspath(__file__))

save = True

#valid_images = ['.svs','.jpg']

def preprocess_input(x0):
    x = x0 / 255.
    x -= 0.5
    x *= 2.
    return x

def samples_normalization (x_data, y_data):
    x_data = x_data.astype('float32')
    mean = np.mean(x_data)  # mean for data centering
    std = np.std(x_data)  # std for data normalization
    x_data -= mean
    x_data /= std
    
    y_data = y_data.astype('float32')
    y_data /= 255.  # scale masks to [0, 1]
    return x_data,y_data,mean,std

def cvpr_2019_normalization (x_data):
    #x_data = x_data.astype('float32')
    max_val = x_data.max()
    min_val = x_data.min()    
    new_x_data = (x_data - min_val)/(max_val-min_val)    
    new_x_data = new_x_data*255
    return new_x_data
    
def split_data_train_val (ac_x_data,x_data,y_data):

    sample_count = len(x_data)   
    train_size = int(sample_count * 4.5 // 5)    
    
    ac_x_train = ac_x_data[:train_size]
    x_train = x_data[:train_size]
    y_train = y_data[:train_size]
    
    ac_x_val = ac_x_data[train_size:]
    x_val = x_data[train_size:]
    y_val = y_data[train_size:]
    
    return ac_x_train,x_train,y_train,ac_x_val,x_val,y_val

def split_data_train_val_cvpr_2019 (ac_x_data,x_data,y_data):

    sample_count = len(x_data)   
    train_size = int(sample_count * 4.5 // 5)    
    
    ac_x_train = ac_x_data[:train_size]
    x_train = x_data[:train_size]
    y_train = y_data[:train_size]
    
    ac_x_val = ac_x_data[train_size:]
    x_val = x_data[train_size:]
    y_val = y_data[train_size:]
    
    return ac_x_train,x_train,y_train,ac_x_val,x_val,y_val
    

# def applyImageAugmentationAndRetrieveGenerator():


#     # We create two instances with the same arguments
#     data_gen_args = dict(rotation_range=90.,
#                          width_shift_range=0.1,
#                          height_shift_range=0.1,
#                          zoom_range=0.2
#                          )
#     image_datagen = ImageDataGenerator(**data_gen_args)
#     mask_datagen = ImageDataGenerator(**data_gen_args)
    
#     # Provide the same seed and keyword arguments to the fit and flow methods
#     seed = 1
    
#     image_generator = image_datagen.flow_from_directory('dataset/train_images',
#                                                         target_size=(360,480),    
#                                                         class_mode=None,
#                                                         seed=seed,
#                                                         batch_size = 32)
    
#     mask_generator = mask_datagen.flow_from_directory('dataset/train_masks',
#                                                       target_size=(360,480),  
#                                                       class_mode=None,
#                                                       seed=seed,
#                                                       batch_size = 32)
    

#     train_generator = zip(image_generator, mask_generator)
    
#     return train_generator

def extract_image_patches(full_img,full_mask,patch_h,patch_w, img_name, imd_saving_dir):
        
    height,width, channel = full_img.shape    
    rows = (int) (height/patch_h)
    columns = (int) (width/patch_w)

    k = 0
    pn = 0
    for r_s in range(rows):
        for c_s in range(columns):
            patch_img = full_img[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w,:]
            patch_mask = full_mask[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w,:]
            f_img_name =str(img_name)+'_'+str(pn)+'.jpg'
            f_mask_name =str(img_name)+'_'+str(pn)+'_mask'+'.jpg'           
            final_des_img = os.path.join(imd_saving_dir,f_img_name)
            final_des_mask = os.path.join(imd_saving_dir,f_mask_name)
            
            mx_val = patch_mask.max()
            mn_val = patch_mask.min()
            
            print ('max_val : '+str(mx_val))
            print ('min_val : '+str(mn_val))          

            if mx_val > 10:
                cv2.imwrite(final_des_img,patch_img)
                cv2.imwrite(final_des_mask,patch_mask)
            pn+=1
            
        k +=1
        print ('Processing for: ' +str(k))

    return pn

def read_single_pixel_anno_data(image_dir,img_h,img_w):

    all_images = [x for x in sorted(os.listdir(image_dir)) if x[-4:] == '.jpg']
    
    total = int(np.round(len(all_images)/2))

    ac_imgs = np.ndarray((total, img_h,img_w,3), dtype=np.uint8)
    imgs = np.ndarray((total, img_h,img_w), dtype=np.uint8)
    imgs_mask = np.ndarray((total,img_h,img_w), dtype=np.uint8)
    k = 0
    print('Creating training images...')
    #img_patients = np.ndarray((total,), dtype=np.uint8)
    for i, image_name in enumerate(all_images):
         if 'mask' in image_name:
             continue
         image_mask_name = image_name.split('.')[0] + '_mask.jpg'
          # patient_num = image_name.split('_')[0]
         img = cv2.imread(os.path.join(image_dir, image_name), cv2.IMREAD_GRAYSCALE)
         ac_img = cv2.imread(os.path.join(image_dir, image_name))
         img_mask = cv2.imread(os.path.join(image_dir, image_mask_name), cv2.IMREAD_GRAYSCALE)
         img_mask = 255.0*(img_mask[:,:]> 0)
         img_mask = cv2.dilate(img_mask,kernel,iterations = 1)
         img_mask = ndimage.gaussian_filter(img_mask, sigma=(1,1),order = 0)     
         img_mask = 255.0*(img_mask[:,:]> 0.3)
        
         
         ac_imgs[k] = ac_img 
         imgs[k] = img
         imgs_mask[k] = img_mask

         k += 1
         print ('Done',i)
     
    """
    perm = np.random.permutation(len(imgs_mask))
    imgs = imgs[perm]
    imgs_mask = imgs_mask[perm]
    ac_imgs = ac_imgs[perm]
    """
    return ac_imgs, imgs, imgs_mask

def create_dataset_patches_driver(image_dir,saving_dir,patch_h,patch_w):
        
    all_images = [x for x in sorted(os.listdir(image_dir)) if x[-4:] == '.bmp']
    

    
    Total_patches = 0

    for i, name in enumerate(all_images):
        
        if 'anno' in name:
            continue
          
        im = cv2.imread(image_dir + name, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
    
        #im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LANCZOS4)

        acc_name = name.split('.')[0]
        mask_name = acc_name +'_anno.bmp'       
        mask_im = cv2.imread(image_dir + mask_name, cv2.IMREAD_UNCHANGED) #.astype('float32')/255.
        #mask_im = cv2.resize(mask_im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
        #y_data[i] = mask_im
        
        mask_im = 255*(mask_im[:,:]>0)
        img_rz = im
        img_mask_rz = mask_im
        
        num_patches = extract_image_patches (img_rz, img_mask_rz, patch_h, patch_w, acc_name, saving_dir)
        
        print ('Processing for: ' +str(i))
        Total_patches = Total_patches + num_patches
    
    return 0


def read_testing_images(data_path,image_h, image_w):
    
    train_data_path = os.path.join(data_path)
    #images = filter((lambda image: 'mask' not in image), os.listdir(train_data_path))
    images = glob.glob(train_data_path + "/*.jpg")
    total = np.round(len(images)) 

    acc_imgs = np.ndarray((total, image_h, image_w,3), dtype=np.uint8)
    gray_mgs = np.zeros((total, image_h, image_w), dtype=np.uint8)
    #imgs_mask = np.zeros((total, image_h, image_w), dtype=np.uint8)
    
    i = 0
    print('Creating training images...')
    #img_patients = np.ndarray((total,), dtype=np.uint8)
    for image_name in images:
        '''
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.jpg'
        '''
        #image_mask_name = image_name.split('/')[-1]      
        #img_first = image_mask_name.split('.')[0]
        #img_second = img_first.split('_mask')[0]      
        #image_name =img_second+'.jpg'
                     
        acc_img = cv2.imread(os.path.join(train_data_path, image_name))
        gray_img = cv2.imread(os.path.join(train_data_path, image_name),cv2.IMREAD_GRAYSCALE)
        #img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
        
        acc_imgs[i] = acc_img
        gray_mgs[i] = gray_img
        #imgs_mask[i] = img_mask

        i += 1
        print ('Done',i)
    
    return acc_imgs



def read_images_and_masks(data_path, image_h, image_w):
    
    train_data_path = os.path.join(data_path)
    images = glob.glob(train_data_path + "/*mask.png")
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
        image_name =img_second+'.jpg'
                     
        acc_img = cv2.imread(os.path.join(train_data_path, image_name))
        img = cv2.imread(os.path.join(train_data_path, image_name),cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
        
        # RBG
#        acc_imgs[i] = acc_img
#        imgs[i] = img
#        imgs_mask[i] = img_mask

                
        # Convert BGR to HSV
        #hsv_img = cv2.cvtColor(acc_img, cv2.COLOR_BGR2HSV)   
        acc_imgs[i] = acc_img
        imgs[i] = img
        imgs_mask[i] = img_mask             

               
        # Convert RGB to YUV
        #yuv_img = cv2.cvtColor(acc_img,cv2.COLOR_RGB2YUV)
        #acc_imgs[i] = yuv_img
        #imgs[i] = img
        #imgs_mask[i] = img_mask  
             
        # Convert RGB to Lab
        #lab_img = cv2.cvtColor(acc_img, cv2.COLOR_BGR2LAB)                
        #acc_imgs[i] = lab_img
        #imgs[i] = img
        #imgs_mask[i] = img_mask  
              
        # RGB to YCrCb
        #ycc_img = cv2.cvtColor(acc_img, cv2.COLOR_BGR2YCR_CB)
        #acc_imgs[i] = ycc_img
        #imgs[i] = img
        #imgs_mask[i] = img_mask  
                
        # RGB to CIE
        #cie_img = cv2.cvtColor(acc_img, cv2.COLOR_BGR2XYZ)
        #acc_imgs[i] = cie_img
        #imgs[i] = img
        #imgs_mask[i] = img_mask  
                
        # RGB to HLS
        #hls_img = cv2.cvtColor(acc_img, cv2.COLOR_BGR2HLS)
        #acc_imgs[i] = hls_img
        #imgs[i] = img
        #imgs_mask[i] = img_mask  
                


        i += 1
        print ('Done',i)
    
    return acc_imgs,imgs,imgs_mask

def read_images_and_masks_RGB(data_path, image_h, image_w):
    
    train_data_path = os.path.join(data_path)
    images = glob.glob(train_data_path + "/*mask.png")
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
        image_name =img_second+'.png'
                     
        acc_img = cv2.imread(os.path.join(train_data_path, image_name))
        img = cv2.imread(os.path.join(train_data_path, image_name),cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
        
        # RBG
        acc_imgs[i] = acc_img
        imgs[i] = img
        imgs_mask[i] = img_mask

        i += 1
        print ('Image reading Done',i)
    
    return acc_imgs,imgs,imgs_mask

def read_images_and_masks_cvpr_2019(data_path, image_h, image_w):
    
    
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
    
def read_traning_data_4classificaiton(base_dir, h,w):
        
    d = defaultdict(list)
    for root, subdirs, files in os.walk(base_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            assert file_path.startswith(base_dir)
            suffix = file_path[len(base_dir):]
            suffix = suffix.lstrip("/")
            label = suffix.split("/")[0]
            d[label].append(file_path)
      
    tags = sorted(d.keys())
    processed_image_count = 0
    useful_image_count = 0

    X = []
    y = []
    
  
    for class_index, class_name in enumerate(tags):
        filenames = d[class_name]
        for filename in filenames:
            processed_image_count += 1         
            img = cv2.imread(filename)
            img_name_1 = filename.split('/')[-1]
            img_name = img_name_1.split('.')[0]
            img_extension = img_name_1.split('.')[1]
            
            if img_extension =='tif':
                img= np.array(img)               
                img = cv2.resize(img, (h,w), interpolation = cv2.INTER_AREA)
                X.append(img)
                y.append(class_index)                
                useful_image_count += 1        

    #pdb.set_trace()   
    X = np.array(X).astype(np.float32)
    X=X.transpose((0,1,2,3))
    X = preprocess_input(X)
    y = np.array(y)

    perm = np.random.permutation(len(y))
    X = X[perm]
    y = y[perm]
    
    print("classes:")
    for class_index, class_name in enumerate(tags):
        print(class_name, sum(y == class_index))
    
    print("\n")

    return X, y, tags

def read_traning_data_4classificaiton_cvpr_2019(base_dir, h,w):
        
    d = defaultdict(list)
    for root, subdirs, files in os.walk(base_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            assert file_path.startswith(base_dir)
            suffix = file_path[len(base_dir):]
            suffix = suffix.lstrip("/")
            label = suffix.split("/")[0]
            d[label].append(file_path)
      
    tags = sorted(d.keys())
    processed_image_count = 0
    useful_image_count = 0

    X = []
    y = []
    
      
    for class_index, class_name in enumerate(tags):
        filenames = d[class_name]
        for filename in filenames:
            processed_image_count += 1         
            img = cv2.imread(filename)
            gray_img = cv2.imread(os.path.join(filename),cv2.IMREAD_GRAYSCALE)
            img_name_1 = filename.split('/')[-1]
            img_name = img_name_1.split('.')[0]
            img_extension = img_name_1.split('.')[1]
            
            if img_extension =='tif':
                img= np.array(img)               
                img = cv2.resize(img, (h,w), interpolation = cv2.INTER_AREA)
                X.append(gray_img)
                y.append(class_index)                
                useful_image_count += 1        

    #pdb.set_trace()  
    X = np.array(X).astype(np.float32)
    X=X.transpose((0,1,2))
    X = preprocess_input(X)
    y = np.array(y)

    perm = np.random.permutation(len(y))
    X = X[perm]
    y = y[perm]
    
    print("classes:")
    for class_index, class_name in enumerate(tags):
        print(class_name, sum(y == class_index))
    
    print("\n")

    return X, y, tags
    
    
def extract_patches_from_image_to_save_in_diretory(full_img,patch_h,patch_w, img_name, patches_saving_dir):
    
    height,width, channel = full_img.shape    
    rows = (int) (height/patch_h)
    columns = (int) (width/patch_w)
    
    k = 0
    pn = 0

     
    for r_s in range(rows):
        for c_s in range(columns):
            patch_img = full_img[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w,:]
            #patch_mask = full_mask[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w,:]
            f_img_name =str(pn)+'.jpg'
            #f_mask_name =str(img_name)+'_'+str(pn)+'_mask'+'.jpg'           
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

def extract_patches_from_image(full_img,patch_h,patch_w):
    
    img_size = full_img.shape 
    
    if len(img_size) > 2:
        height = img_size[0]
        width = img_size [1]
        channels = img_size [2]
    else:
        height = img_size[0]
        width = img_size [1]
   
    rows = (int) (height/patch_h)
    columns = (int) (width/patch_w)
    pn = 0    
    patches = []       
    for r_s in range(rows):
        for c_s in range(columns):            
            # extract non_overlapping patches...
            if len(img_size) > 2:
                idv_patch_img = full_img[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w,:]
            else:
                idv_patch_img = full_img[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w]
            
            patches.append(idv_patch_img)
            pn+=1
       
    patches = np.array(patches).astype(np.float32) 
    
    return patches,pn,rows,columns


def extract_patches_from_svs(svs_img_dir, patches_saving_dir, patch_size):
            
        
    if not os.path.isdir("%s/%s"%(patches_saving_dir,"patches")):
        os.makedirs("%s/%s"%(patches_saving_dir,"patches"))
    
    patch_dir_name = 'patches_'+str(patch_size[0])+'/'
    
    patches_dir = join_path(patches_saving_dir+patch_dir_name)
    
    image_svs = [x for x in sorted(os.listdir(svs_img_dir)) if x[-4:] == '.svs']
    
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
        
        # create an array to store our image
        #img_np = np.zeros((orig_w,orig_h,3),dtype=np.uint8)
        
        #pdb.set_trace()
        
        
        img_saving_idx = 0
        
        for r in range(0,orig_w,patch_size[0]):
            for c in range(0, orig_h,patch_size[1]):
                
                if c+patch_size[1] > orig_h and r+patch_size[0]<= orig_w:
                    p = orig_h-c
                    img = np.array(scan.read_region((c,r),0,(p,patch_size[1])),dtype=np.uint8)[...,0:3]
                elif c+patch_size[1] <= orig_h and r+patch_size[0] > orig_w:
                    p = orig_w-r
                    img = np.array(scan.read_region((c,r),0,(patch_size[0],p)),dtype=np.uint8)[...,0:3]
                elif  c+patch_size[1] > orig_h and r+patch_size[0] > orig_w:
                    p = orig_h-c
                    pp = orig_w-r
                    img = np.array(scan.read_region((c,r),0,(p,pp)),dtype=np.uint8)[...,0:3]
                else:    
                    img = np.array(scan.read_region((c,r),0,(patch_size[0],patch_size[1])),dtype=np.uint8)[...,0:3]
                    
                           
                print("Processing:"+str(img_saving_idx))                
                ac_img_name =str(img_saving_idx)+'.jpg'
                final_img_des = os.path.join(patches_sub_dir,ac_img_name)
                #cv2.imwrite(final_img_des,img)
                imsave(final_img_des,img)                
                img_saving_idx +=1
      
        scan.close


def patches_to_image_log_from_dir(patches_dir):
    
    image_id = []
    image_h = []
    image_w = []
    patch_h = []
    patch_w = []
    
   
    json_files = [x for x in sorted(os.listdir(patches_dir)) if x[-5:] == '.json']           
    image_files = [x for x in sorted(os.listdir(patches_dir)) if x[-4:] == '.jpg']
    
    
    names_wo_ext = []
    for idx in range(len(image_files)):
        name = image_files[idx]
        name_wo_ext = int(name.split('.')[0])
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
        image_h = image_logs["height"]
        image_w = image_logs["width"]
        patch_w = image_logs["patch_width"]
        patch_h = image_logs["patch_height"]        
        num_rows = image_logs["no_patches_x_axis"]
        num_columns = image_logs["no_patches_y_axis"]
            
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
            
            img_from_patches[r*patch_w: r*patch_w+patch_w,c*patch_h:c*patch_h+patch_h,:] = patch
            
            print("Merging patch no. :"+str(patch_idx))               
            patch_idx +=1
    
    img_name =str(image_id)+'_merge.jpg'
    final_img_des = os.path.join(patches_dir,img_name)
    imsave(final_img_des,img_from_patches)
    

def image_from_patches(patches,num_patches, num_rows, num_columns):
    
    patches_size = patches.shape
    
    patch_w = patches_size[1]
    patch_h = patches_size[2]
    
    image_w = patches_size[1]*num_rows
    image_h = patches_size[2]*num_columns
    
    if len(patches_size)>3:
        img_from_patches = np.zeros((image_w,image_h,3),dtype=np.uint8)
    else:
        img_from_patches = np.zeros((image_w,image_h),dtype=np.uint8)
    
    patch_idx = 0    
    
    for r in range(0,num_rows):
        for c in range(0, num_columns):

            if len(patches_size)>3:
                img_from_patches[r*patch_w: r*patch_w+patch_w,c*patch_h:c*patch_h+patch_h,:] = patches[patch_idx]
            else:
                img_from_patches[r*patch_w: r*patch_w+patch_w,c*patch_h:c*patch_h+patch_h] = patches[patch_idx]
            
            patch_idx +=1
    
    img_from_patches = np.array(img_from_patches).astype(np.float32) 
     
    return  img_from_patches  
    
    
def image2patches_driver(patches_source, patch_size):
    
    #pdb.set_trace()
    
    dir_name = "patches_"+str(patch_size[0])
    if not os.path.isdir("%s/%s"%(patches_saving_dir,dir_name)):
        os.makedirs("%s/%s"%(patches_saving_dir,dir_name))
    
    patch_dir_name = 'patches_'+str(patch_size[0])+'/'
    
    patches_dir = join_path(patches_saving_dir,patch_dir_name)
    
    image_jpg= [x for x in sorted(os.listdir(patches_source)) if x[-4:] == '.jpg']
   
    #for f in os.listdir(image_svs):
    for i, f in enumerate(image_jpg):
        
        ext = os.path.splitext(f)[1]

        if ext.lower() not in allowed_extensions:
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
        
        patches, patches_number = extract_patches_from_image(img,patch_size[0],patch_size[1])
   
        print(str(patches_number))

    return patches_sub_dir,patches, patches_number
    

def image_and_mask2patches_driver(patches_source, patches_saving_dir,patch_size):
  
    take_dir_name = patches_source.split('/')[-2]
    
    dir_name = take_dir_name
    
    if not os.path.isdir("%s/%s"%(patches_saving_dir,dir_name)):
        os.makedirs("%s/%s"%(patches_saving_dir,dir_name))
    
    patch_dir_name = 'img_mask_patches'+str(patch_size[0])+'/'    
    patches_dir = join_path(patches_saving_dir,patch_dir_name)
    
    if not os.path.isdir("%s/%s"%(patches_dir,"images")):
        os.makedirs("%s/%s"%(patches_saving_dir,"images"))
    
    img_patch_dir_name = 'images'+'/'    
    img_patches_dir = join_path(patches_saving_dir,img_patch_dir_name)
    
    if not os.path.isdir("%s/%s"%(patches_dir,"masks")):
        os.makedirs("%s/%s"%(patches_saving_dir,"masks"))
    
    mask_patch_dir_name = 'masks'+'/'    
    mask_patches_dir = join_path(patches_saving_dir,mask_patch_dir_name)
    
    image_jpg= [x for x in sorted(os.listdir(patches_source)) if x[-4:] == '.jpg']
    #images = filter((lambda image: 'mask' not in image), os.listdir(train_data_path))
    #for f in os.listdir(image_svs):
    for i, img_file_name in enumerate(image_jpg):
        
        if 'mask' in img_file_name:
            continue
        
        if ext.lower() not in allowed_extensions:
            continue
        
        ext = os.path.splitext(img_file_name)[1] 
        
        dir_name = os.path.splitext(img_file_name)[0]       
        img_name = img_name
        mask_name = img_name+'_mask'+ext
        
        print(patches_source.split('/')[0])
        
        if not os.path.isdir("%s/%s"%(patches_dir,dir_name)):
            os.makedirs("%s/%s"%(patches_dir,dir_name))
        
        patches_sub_dir = join_path(patches_dir+dir_name+'/')
        # open scan
        img_path = os.path.join(patches_source,img_file_name)
        mask_path = os.path.join(patches_source,mask_name)
        
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
        mask_im = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) #.astype('float32')/255.
        
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
        # extract and save patches for images..
        patches_number = extract_patches_from_image(img,patch_size[0],patch_size[1], img_name, img_patches_dir)
        # extract and save patches for images..
        patches_number = extract_patches_from_image(mask_im,patch_size[0],patch_size[1], img_name, img_patches_dir)
   
        print(str(patches_number))

    return patches_sub_dir