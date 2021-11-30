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
#from skimage.transform import resize
#import shutil
import scipy.ndimage as ndimage
from scipy import stats
from collections import Counter

kernel = np.ones((7,7), np.uint8) 

allowed_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp','*.mat','*.tif']

#abspath = os.path.dirname(os.path.abspath(__file__))

save = True

#valid_images = ['.svs','.jpg']
def flipRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def preprocess_input(x0):
    x = x0 / 255.
    x -= 0.5
    x *= 2.
    return x

def preprocess_input_nash(x0):
    x = x0 / 255.
    #x -= 0.5
    #x *= 2.
    return x

def preprocess_input_steatosis(x0):
    x = x0 / 255.
    #x -= 0.5
    #x *= 2.
    return x

def preprocess_input_inflm(x0):
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


def applyImageAugmentationAndRetrieveGenerator():


    # We create two instances with the same arguments
    data_gen_args = dict(rotation_range=90.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2
                         )
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    
    image_generator = image_datagen.flow_from_directory('dataset/train_images',
                                                        target_size=(360,480),    
                                                        class_mode=None,
                                                        seed=seed,
                                                        batch_size = 32)
    
    mask_generator = mask_datagen.flow_from_directory('dataset/train_masks',
                                                      target_size=(360,480),  
                                                      class_mode=None,
                                                      seed=seed,
                                                      batch_size = 32)
    

    train_generator = zip(image_generator, mask_generator)
    
    return train_generator

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
    images = glob.glob(train_data_path + "/*mask.jpg")
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
        
        acc_imgs[i] = acc_img
        imgs[i] = img
        imgs_mask[i] = img_mask

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
    
    #pdb.set_trace()    
    for class_index, class_name in enumerate(tags):
        filenames = d[class_name]
        for filename in filenames:
            processed_image_count += 1         
            img = cv2.imread(filename)
            img_name_1 = filename.split('/')[-1]
            img_name = img_name_1.split('.')[0]
            img_extension = img_name_1.split('.')[1]
            
            if img_extension =='jpg':
                img= np.array(img)               
                img = cv2.resize(img, (h,w), interpolation = cv2.INTER_AREA)
                X.append(img)
                y.append(class_index)                
                useful_image_count += 1        

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

def extract_patches_from_image_mask(full_img,gray_mask, patch_h,patch_w):
    
    img_size = full_img.shape 
    
    ret,mask = cv2.threshold(gray_mask,127,255,cv2.THRESH_BINARY)

    
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
    start_x = []
    start_y = []
    patches = []     
    
    #pdb.set_trace()
    
    for r_s in range(rows):
        for c_s in range(columns):            
            # extract non_overlapping patches...
            if len(img_size) > 2:
                indv_mask = mask[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w]
                
                #image_name = 'test.jpg' 
                #cv2.imwrite(image_name,indv_mask)
            
                indv_mask = 1*(indv_mask[:,:]>50)
                sum_pixels = indv_mask.sum()
                if sum_pixels > 500:
                    idv_patch_img = full_img[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w,:]
                    start_x.append(r_s)
                    start_y.append(c_s)
                    patches.append(idv_patch_img)            
            pn+=1
       
    patches = np.array(patches).astype(np.float32) 
    
    return patches,pn,rows,columns,start_x,start_y

def extract_contour_centers(pred_mask,mitosis_width,mitosis_height):         
      
    cntr_points_x = []
    cntr_points_y = []    
    temp_bin_img = pred_mask.copy()    
    temp_bin_img = temp_bin_img.astype('uint8')
    im2, contours, hierarchy = cv2.findContours(temp_bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    num_regions = len(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:num_regions] # get largest five contour area
    
    patches_size = pred_mask.shape
        
    width = patches_size[0]
    height = patches_size[1]
        
    cntr_points_x = []
    cntr_points_y = [] 
        
    for i in range(len(contours)):
        M = cv2.moments(contours[i])
        if(M['m00'] == 0.0):
            continue
        x, y = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])            
        #print("x and y coordinate for y_hat",x,y)            
        if y < int(mitosis_width/2) or x < int(mitosis_height/2) or y > int(width-mitosis_width/2) or x > int(height-mitosis_height/2):
            if x < int(mitosis_height/2) and y < int(mitosis_width/2):
                cntr_points_x.append(int(mitosis_width/2))
                cntr_points_y.append(int(mitosis_height/2))   
                        
            if x < int(mitosis_height/2) and y > int(mitosis_width/2):
                if y > int(width-mitosis_width/2):
                    cntr_points_x.append(int(mitosis_height/2))
                    cntr_points_y.append(int(width-mitosis_width/2))
                else:
                    cntr_points_x.append(int(mitosis_height/2))
                    cntr_points_y.append(int(y))
                    
            if x > int(mitosis_height/2) and y < int(mitosis_width/2):
                if x > int(height-mitosis_height/2):
                    cntr_points_x.append(int(height-mitosis_height/2))
                    cntr_points_y.append(int(mitosis_width/2))
                else:                        
                    cntr_points_x.append(int(x))
                    cntr_points_y.append(int(mitosis_width/2))
                    
            if x > int(height-mitosis_height/2) and y > int(mitosis_width/2):                    
                if y > int(width-mitosis_width/2):
                    cntr_points_x.append(int(height-mitosis_height/2))
                    cntr_points_y.append(int(width-mitosis_width/2))
                else:                   
                    cntr_points_x.append(int(height-mitosis_height/2))
                    cntr_points_y.append(int(y))                    
                        
            if x > int(mitosis_height/2) and y > int(width-mitosis_width/2):
                if x > int(height-mitosis_height/2):
                    cntr_points_x.append(int(height-mitosis_height/2))
                    cntr_points_y.append(int(width-mitosis_width/2))
                else:
                    cntr_points_x.append(int(x))
                    cntr_points_y.append(int(width-mitosis_width/2))
        else:            
            cntr_points_x.append(int(x))
            cntr_points_y.append(int(y))
            
    return cntr_points_x,cntr_points_y
    
def extract_patches_from_image_mask_testing(full_img,gray_mask, patch_h,patch_w):
    
    img_size = full_img.shape     
    ret,mask = cv2.threshold(gray_mask,127,255,cv2.THRESH_BINARY)    
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
    start_x = []
    start_y = []
    patches = []         
    #pdb.set_trace()    
    for r_s in range(rows):
        for c_s in range(columns):            
            # extract non_overlapping patches...
            if len(img_size) > 2:
                indv_mask = mask[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w]                
                #image_name = 'test.jpg' 
                #cv2.imwrite(image_name,indv_mask)            
                indv_mask = 1*(indv_mask[:,:]>100)
                sum_pixels = indv_mask.sum()
                if sum_pixels > 0:
                    idv_patch_img = full_img[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w,:]
                    start_x.append(r_s)
                    start_y.append(c_s)
                    patches.append(idv_patch_img)            
            pn+=1       
    patches = np.array(patches).astype(np.float32)     
    return patches,pn,rows,columns,start_x,start_y
    
    
def generate_heatmap_image_from_conf_values(logs, patch_h, patch_w, HPF_height, HPF_width):
    
    heat_maps_mat = np.zeros((HPF_height,HPF_width),dtype=int)
    
    num_samples_logs = np.array(logs)
    
    for k in range(num_samples_logs.shape[0]):
        
        single_log = num_samples_logs[k,:]
        row = int(single_log[0]*patch_h)
        column = int(single_log[1]*patch_w)
        conf_value = single_log[2]
        index_value = int(single_log[3]+1)
        
        heat_maps_mat [row:row+patch_h, column: column+patch_w] = int(index_value)
    
    
    return heat_maps_mat
    
def generate_heatmap_image_from_conf_values_on_org(image, logs, patch_h, patch_w, HPF_height, HPF_width):
    
    #heat_maps_mat = np.zeros((HPF_height,HPF_width),dtype=int)    
    num_samples_logs = np.array(logs)    
    for k in range(num_samples_logs.shape[0]):        
        single_log = num_samples_logs[k,:]
        #print(single_log)
        row = int(single_log[0]*patch_h)
        column = int(single_log[1]*patch_w)
        conf_value = single_log[2]
        index_value = int(single_log[3])
        
        #pdb.set_trace()
        
        if index_value == 0:
            image [row:row+patch_h, column: column+patch_w, :] = 125
        else:
            image [row:row+patch_h, column: column+patch_w, :] = 0
    
    heat_maps_mat = image
    
    return heat_maps_mat

def generate_heatmap_image_from_conf_values_on_org_saving_FP(image, logs, patch_h, patch_w, HPF_height, HPF_width, img_name_wo_ext, FP_saving_path):
    
    #heat_maps_mat = np.zeros((HPF_height,HPF_width),dtype=int)  
    img_id = 0
    
    num_samples_logs = np.array(logs)    
    for k in range(num_samples_logs.shape[0]):        
        single_log = num_samples_logs[k,:]
        #print(single_log)
        row = int(single_log[0]*patch_h)
        column = int(single_log[1]*patch_w)
        conf_value = single_log[2]
        index_value = int(single_log[3])
        
        #pdb.set_trace()
        
        if index_value == 0:
            image [row:row+patch_h, column: column+patch_w, :] = 125
        else:
            indv_patch = image [row:row+patch_h, column: column+patch_w, :]            
            image_name = img_name_wo_ext+'_'+str(img_id)+'.jpg' 
            final_des_image = os.path.join(FP_saving_path,image_name)
            cv2.imwrite(final_des_image,indv_patch)
            
            img_id = img_id+1
    
            image [row:row+patch_h, column: column+patch_w, :] = 0
    
    heat_maps_mat = image
    
    return heat_maps_mat
    

def generate_final_mask_from_seg_class(mask, logs, patch_h, patch_w, HPF_height, HPF_width):
    
    #final_mask = mask  
    #per_patch_pixels = 96*96
    final_mask = np.zeros((HPF_height,HPF_width),dtype=int)
    img_id = 0
    #num_pxls_bronchial_cells_HPFs = 0
    #num_pxls_lyphoctyte_cells_HPFs = 0
    num_samples_logs = np.array(logs)    
    for k in range(num_samples_logs.shape[0]):        
        single_log = num_samples_logs[k,:]
        row = int(single_log[0]*patch_h)
        column = int(single_log[1]*patch_w)
        conf_value = single_log[2]
        index_value = int(single_log[3])
        
        if index_value == 0:
            #final_mask [row:row+patch_h, column: column+patch_w] = mask [row:row+patch_h, column: column+patch_w]
            indv_mask = mask [row:row+patch_h, column: column+patch_w]
            indv_mask = (indv_mask/255)*125              
            final_mask [row:row+patch_h, column: column+patch_w ] = indv_mask            
            #num_pxls_bronchial_cells_HPFs = num_pxls_bronchial_cells_HPFs + per_patch_pixels
        else:
            final_mask [row:row+patch_h, column: column+patch_w] = mask [row:row+patch_h, column: column+patch_w]   
            #num_pxls_lyphoctyte_cells_HPFs = num_pxls_lyphoctyte_cells_HPFs + per_patch_pixels
    
    return final_mask #, num_pxls_bronchial_cells_HPFs, num_pxls_lyphoctyte_cells_HPFs

def generate_final_mask_from_seg_five_class(mask, logs, patch_h, patch_w, HPF_height, HPF_width):
    
    #final_mask = mask  
    #per_patch_pixels = 96*96
    final_mask = np.zeros((HPF_height,HPF_width),dtype=int)
    img_id = 0
    #num_pxls_bronchial_cells_HPFs = 0
    #num_pxls_lyphoctyte_cells_HPFs = 0
    num_samples_logs = np.array(logs)
    for k in range(num_samples_logs.shape[0]):        
        single_log = num_samples_logs[k,:]
        row = int(single_log[0]*patch_h)
        column = int(single_log[1]*patch_w)
        conf_value = single_log[2]
        index_value = int(single_log[3])
        
        if index_value == 0:  #Bronchiacell(Green)
            #final_mask [row:row+patch_h, column: column+patch_w] = mask [row:row+patch_h, column: column+patch_w]
            indv_mask = mask [row:row+patch_h, column: column+patch_w]
            indv_mask = (indv_mask/255)*130              
            final_mask [row:row+patch_h, column: column+patch_w ] = indv_mask            
            #num_pxls_bronchial_cells_HPFs = num_pxls_bronchial_cells_HPFs + per_patch_pixels
        elif index_value == 1:   #(Lymphcyte)  (red)
            final_mask [row:row+patch_h, column: column+patch_w] = mask [row:row+patch_h, column: column+patch_w]         
        elif index_value == 2:  # macrophase   (light green)
            #final_mask [row:row+patch_h, column: column+patch_w] = mask [row:row+patch_h, column: column+patch_w]
            indv_mask = mask [row:row+patch_h, column: column+patch_w]
            indv_mask = (indv_mask/255)*190              
            final_mask [row:row+patch_h, column: column+patch_w ] = indv_mask  
        elif index_value == 3:   # RBC  close ro read...
            #final_mask [row:row+patch_h, column: column+patch_w] = mask [row:row+patch_h, column: column+patch_w]
            indv_mask = mask [row:row+patch_h, column: column+patch_w]
            indv_mask = (indv_mask/255)*80              
            final_mask [row:row+patch_h, column: column+patch_w ] = indv_mask  
        else:
            #final_mask [row:row+patch_h, column: column+patch_w] = mask [row:row+patch_h, column: column+patch_w]  
            indv_mask = mask [row:row+patch_h, column: column+patch_w]
            indv_mask = (indv_mask/255)*0              
            final_mask [row:row+patch_h, column: column+patch_w ] = indv_mask 
            #num_pxls_lyphoctyte_cells_HPFs = num_pxls_lyphoctyte_cells_HPFs + per_patch_pixels
    
    return final_mask #, num_pxls_bronchial_cells_HPFs, num_pxls_lyphoctyte_cells_HPFs    


def generate_final_mask_from_steatosis_seg_class(mask, logs, patch_h, patch_w, HPF_height, HPF_width):
    
    #final_mask = mask  
    #per_patch_pixels = 96*96
    final_mask = np.zeros((HPF_height,HPF_width),dtype=int)
    img_id = 0
    #num_pxls_bronchial_cells_HPFs = 0
    #num_pxls_lyphoctyte_cells_HPFs = 0
    num_samples_logs = np.array(logs)
    for k in range(num_samples_logs.shape[0]):        
        single_log = num_samples_logs[k,:]
        row = int(single_log[0]*patch_h)
        column = int(single_log[1]*patch_w)
        conf_value = single_log[2]
        index_value = int(single_log[3])
        
        if index_value == 0:  #Bronchiacell(Green)
            #final_mask [row:row+patch_h, column: column+patch_w] = mask [row:row+patch_h, column: column+patch_w]
            indv_mask = mask [row:row+patch_h, column: column+patch_w]
            #indv_mask = (indv_mask/255)*130              
            final_mask [row:row+patch_h, column: column+patch_w ] = indv_mask            
            #num_pxls_bronchial_cells_HPFs = num_pxls_bronchial_cells_HPFs + per_patch_pixels
        elif index_value == 1:   #(Lymphcyte)  (red)
            final_mask [row:row+patch_h, column: column+patch_w] = mask [row:row+patch_h, column: column+patch_w]         
        elif index_value == 2:  # macrophase   (light green)
            #final_mask [row:row+patch_h, column: column+patch_w] = mask [row:row+patch_h, column: column+patch_w]
            indv_mask = mask [row:row+patch_h, column: column+patch_w]
            #indv_mask = (indv_mask/255)*190              
            final_mask [row:row+patch_h, column: column+patch_w ] = indv_mask  
        elif index_value == 3:   # RBC  close ro read...
            #final_mask [row:row+patch_h, column: column+patch_w] = mask [row:row+patch_h, column: column+patch_w]
            indv_mask = mask [row:row+patch_h, column: column+patch_w]
            #indv_mask = (indv_mask/255)*80              
            final_mask [row:row+patch_h, column: column+patch_w ] = indv_mask  
        else:
            #final_mask [row:row+patch_h, column: column+patch_w] = mask [row:row+patch_h, column: column+patch_w]  
            indv_mask = mask [row:row+patch_h, column: column+patch_w]
            indv_mask = (indv_mask/255)*0              
            final_mask [row:row+patch_h, column: column+patch_w ] = indv_mask 
            #num_pxls_lyphoctyte_cells_HPFs = num_pxls_lyphoctyte_cells_HPFs + per_patch_pixels
    
    return final_mask #, num_pxls_bronchial_cells_HPFs, num_pxls_lyphoctyte_cells_HPFs    

def generate_final_mask_from_steatosis_seg_class_with_logs(mask, start_x,start_y,max_conf_values,class_indexes, patch_h, patch_w, HPF_height, HPF_width):
    
    #final_mask = mask  
    #per_patch_pixels = 96*96
    final_mask = np.zeros((HPF_height,HPF_width),dtype=int)
    #img_id = 0
    #num_pxls_bronchial_cells_HPFs = 0
    #num_pxls_lyphoctyte_cells_HPFs = 0
    #num_samples_logs = np.array(logs)
    for k in range(len(start_x)):        
        #single_log = num_samples_logs[k,:]
        row = int(start_x[k]*patch_h)
        column = int(start_y[k]*patch_w)
        #conf_value = max_conf_values[k]
        index_value = int(class_indexes[k])
        
        if index_value <= 3:  #Bronchiacell(Green)
            #final_mask [row:row+patch_h, column: column+patch_w] = mask [row:row+patch_h, column: column+patch_w]
            indv_mask = mask [row:row+patch_h, column: column+patch_w]
            #indv_mask = (indv_mask/255)*130              
            final_mask [row:row+patch_h, column: column+patch_w ] = indv_mask            
            #num_pxls_bronchial_cells_HPFs = num_pxls_bronchial_cells_HPFs + per_patch_pixels
#        elif index_value == 1:   #(Lymphcyte)  (red)
#            final_mask [row:row+patch_h, column: column+patch_w] = mask [row:row+patch_h, column: column+patch_w]         
#        elif index_value == 2:  # macrophase   (light green)
#            #final_mask [row:row+patch_h, column: column+patch_w] = mask [row:row+patch_h, column: column+patch_w]
#            indv_mask = mask [row:row+patch_h, column: column+patch_w]
#            #indv_mask = (indv_mask/255)*190              
#            final_mask [row:row+patch_h, column: column+patch_w ] = indv_mask  
#        elif index_value == 3:   # RBC  close ro read...
#            #final_mask [row:row+patch_h, column: column+patch_w] = mask [row:row+patch_h, column: column+patch_w]
#            indv_mask = mask [row:row+patch_h, column: column+patch_w]
#            #indv_mask = (indv_mask/255)*80              
#            final_mask [row:row+patch_h, column: column+patch_w ] = indv_mask  
        else:
            #final_mask [row:row+patch_h, column: column+patch_w] = mask [row:row+patch_h, column: column+patch_w]  
            indv_mask = mask [row:row+patch_h, column: column+patch_w]
            indv_mask = (indv_mask/255)*0              
            final_mask [row:row+patch_h, column: column+patch_w ] = indv_mask 
            #num_pxls_lyphoctyte_cells_HPFs = num_pxls_lyphoctyte_cells_HPFs + per_patch_pixels
    
    return final_mask #, num_pxls_bronchial_cells_HPFs, num_pxls_lyphoctyte_cells_HPFs    

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
    

def generate_final_mask_from_seg_class_v2(mask, logs, patch_h, patch_w, HPF_height, HPF_width):
    
    #final_mask = mask  
    #per_patch_pixels = 96*96
    final_mask = np.zeros((HPF_height,HPF_width),dtype=int)
    img_id = 0
    #num_pxls_bronchial_cells_HPFs = 0
    #num_pxls_lyphoctyte_cells_HPFs = 0
    num_samples_logs = np.array(logs)    
    for k in range(num_samples_logs.shape[0]):        
        single_log = num_samples_logs[k,:]
        row = int(single_log[0]*patch_h)
        column = int(single_log[1]*patch_w)
        conf_value = single_log[2]
        index_value = int(single_log[3])
        
        if index_value == 0:
            #final_mask [row:row+patch_h, column: column+patch_w] = mask [row:row+patch_h, column: column+patch_w]
            indv_mask = mask [row:row+patch_h, column: column+patch_w]
            indv_mask = (indv_mask/255)*125              
            final_mask [row:row+patch_h, column: column+patch_w ] = indv_mask            
            #num_pxls_bronchial_cells_HPFs = num_pxls_bronchial_cells_HPFs + per_patch_pixels
        else:
            row_ref_pixel = final_mask[row-2:row,column: column+patch_w]
            column_ref_pixel = final_mask[row:row+patch_h,column-2: column]
            
            row_ref_pixel = Counter(row_ref_pixel.flatten()) 
            get_mode_row = dict(row_ref_pixel) 
            mode_row = [k for k, v in get_mode_row.items() if v == max(list(row_ref_pixel.values()))]   
            
            column_ref_pixel = Counter(column_ref_pixel.flatten()) 
            get_mode_col = dict(column_ref_pixel) 
            mode_col = [k for k, v in get_mode_col.items() if v == max(list(column_ref_pixel.values()))]  
            
            if (mode_row[0] ==125) or (mode_col_row ==125):
                indv_mask = mask [row:row+patch_h, column: column+patch_w]
                indv_mask = (indv_mask/255)*125
                final_mask [row:row+patch_h, column: column+patch_w ] = indv_mask 
            else:
                final_mask [row:row+patch_h, column: column+patch_w] = mask [row:row+patch_h, column: column+patch_w]   
            #num_pxls_lyphoctyte_cells_HPFs = num_pxls_lyphoctyte_cells_HPFs + per_patch_pixels
            #final_mask [row:row+patch_h, column: column+patch_w] = mask [row:row+patch_h, column: column+patch_w]   

    
    return final_mask #, num_pxls_bronchial_cells_HPFs, num_pxls_lyphoctyte_cells_HPFs

def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
 
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
 
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
 
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
 
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)
    
def refined_final_heatmap_image(mask_img, mask_seg_clas):
    
    #refined_mask = np.zeros((HPF_height,HPF_width),dtype=int)  
    refined_mask = mask_seg_clas.copy()
    
    mask_img = mask_img.astype('uint8')
    (_, contours, _) = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
    #cntsSorted = sorted(contours, key=lambda x: cv2.cobountourArea(x))    
    #(contours_4avg, boundingBoxes) = sort_contours(contours, method="left-to-right")
    #contours_4avg = sorted(contours, key=cv2.contourArea, reverse=True)
    
    #first_area = contours_4avg[0][0]
    #last_area = contours_4avg[-1]
    #average_area = (first_area+last_area)/2
    average_area = 500
    
    for c in contours:
        if cv2.contourArea(c) > average_area:
            #continue
            # detected
            (x, y, w, h) = cv2.boundingRect(c)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            regional_idv_region = mask_seg_clas[x:x + w,y:y + h]
            regional_idv_region= np.array(regional_idv_region)
            
            refined_mask_individual = np.ones((w,h),dtype=int) 
            #uniques_values = np.unique(generated_mask_seg_clas)
            #print(uniques_values)  
            regional_idv_region.flatten()
            data = Counter(regional_idv_region.flatten()) 
            get_mode = dict(data) 
            mode = [k for k, v in get_mode.items() if v == max(list(data.values()))]               
                                   
            print('Mode is : '+str(mode[0])) 
            if mode[0] > 0:
                final_indv_mask = refined_mask_individual*mode[0]
            else:
                final_indv_mask = refined_mask_individual
            
            refined_mask[x:x + w,y:y + h] = final_indv_mask
   
    return refined_mask