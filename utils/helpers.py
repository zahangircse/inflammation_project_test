import cv2
import numpy as np
import itertools
import operator
import os, csv
import tensorflow as tf
import scipy.ndimage as ndimage
import time, datetime
from os.path import join as join_path

kernel = np.ones((5,5), np.uint8) 

font = cv2.FONT_HERSHEY_SIMPLEX
import pdb
from skimage.util import img_as_ubyte
#import prepered_cvrp_2019_database_demo as cvpr_2019_data_demo

def get_label_info(csv_path):
    """
    Retrieve the class names and label values for the selected dataset.
    Must be in CSV format!

    # Arguments
        csv_path: The file path of the class dictionairy
        
    # Returns
        Two lists: one for the class names and the other for the label values
    """
    filename, file_extension = os.path.splitext(csv_path)
    if not file_extension == ".csv":
        return ValueError("File is not a CSV!")

    class_names = []
    label_values = []
    with open(csv_path, 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        header = next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            label_values.append([int(row[1]), int(row[2]), int(row[3])])
        # print(class_dict)
    return class_names, label_values

def extract_patches(single_image_for_test,cntr_points_x,cntr_points_y,num_test_image,patch_h,patch_w):
    
    patches_for_test = np.ndarray((num_test_image, patch_h, patch_w,3), dtype=np.uint8)
        
    k = 0
    for x,y in zip(cntr_points_x,cntr_points_y):
        xmin = int(x - patch_h/2)
        ymin = int(y - patch_w/2)
        xmax = int(x + patch_h/2)
        ymax = int(y + patch_w/2)                  
        patches_for_test[k,:,:,:] = single_image_for_test[xmin:xmax,ymin:ymax]              
        k += 1
    return patches_for_test

'''
def images_with_confidents(image, points_x,points_y, confs):
    
    image_h, image_w,_ = image.shape
    
    total = len(points_x)
    for x,y,c in zip(points_y,points_x,confs):
       
        cv2.putText(image,str(c),(int(x),int(y)), font,0.3,(255,0,0))       
        # Draw the rectagle over the area..
        cv2.rectangle(image,(x-15,y-15),(x+15,y+15),(255,0,0),2)
        #cv2.putText(image,'x',(int(x),int(y)))#,cv2.FONT_HERSHEY_SIMPLEX,(0,255,0),2)
        """
        cv2.putText(image, 
                    labels[box.get_label()] + ' ' + str(box.get_score()), 
                    (xmin, ymin - 13), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1e-3 * image_h, 
                    (0,255,0), 2)
        """
    
    cv2.putText(image,'Total: '+str(total),(int(5),int(10)), font,0.3,(255,0,0))    
    
    return image 
'''
def blobs_detector(pred):
    
    r,c = pred.shape       
    pred[0:1,:]=0
    pred[r-2,:]=0    
    pred[:,0:1]=0
    pred[:,c-2]=0     
    #pred_erosion = cv2.erode(pred, kernel, iterations=1)   
    params = cv2.SimpleBlobDetector_Params()
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 5
    detector = cv2.SimpleBlobDetector_create(params)
    
    pred_reverse=255-(255*pred)  
    blobs_centers_pred = detector.detect(pred_reverse)
    
    return blobs_centers_pred
    
def perform_morphological_operations(pred):
    
    #pred_mask = cv2.erode(pred,kernel,iterations = 1)

    pred_mask = cv2.dilate(pred,kernel,iterations = 1)            
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)    #  To fillup the internal pixels...
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)     # opening operation to remove the noise 
    pred_mask = cv2.erode(pred_mask,kernel,iterations = 1)

    #pred_mask = cv2.dilate(pred_mask,kernel,iterations = 1)       
    #pred_mask = cv2.erode(pred_mask,kernel,iterations = 1)  
    #pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
     
    pred = pred_mask        
    r,c = pred_mask.shape       
    #pred_mask[0:1,:]=0
    #pred_mask[r-1:r,:]=0    
    #pred_mask[:,0:2]=0
    #pred_mask[:,c-2:c]=0 
    
    return pred_mask

def blobs_filterByCircularity(binary_img):
    #binary_img = img_as_ubyte(binary_img)
    #binary_img = binary_img.astype('int8')
    binary_img = (binary_img/255).astype('uint8')
    #binary_img=cv2.bitwise_not(binary_img)
    #binary_img=255-(255*pred)  
    params = cv2.SimpleBlobDetector_Params()
    # Disable unwanted filter criteria params
    params.filterByInertia = False
    params.filterByConvexity = True
    params.filterByCircularity = True
    #params.minCircularity = 0.1
    detector = cv2.SimpleBlobDetector_create(params)
    # Detect blobs.
    keypoints = detector.detect(binary_img)
    #binary_img=cv2.bitwise_not(binary_img)
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(binary_img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return im_with_keypoints

def perform_morphological_operations_steatosis_seg(pred):
    
    #pred_mask = cv2.erode(pred,kernel,iterations = 1)

    pred_mask = cv2.dilate(pred,kernel,iterations = 1)            
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)    #  To fillup the internal pixels...
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)     # opening operation to remove the noise 
    pred_mask = cv2.erode(pred_mask,kernel,iterations = 1)

    #pred_mask = cv2.dilate(pred_mask,kernel,iterations = 1)       
    #pred_mask = cv2.erode(pred_mask,kernel,iterations = 1)  
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
     
    #pred = pred_mask        
#    r,c = pred_mask.shape       
#    pred_mask[0:2,:]=0
#    pred_mask[r-2:r,:]=0    
#    pred_mask[:,0:2]=0
#    pred_mask[:,c-2:c]=0 
    
    return pred_mask

def extract_patches_for_classification(img_4_patches, cntr_points_x,cntr_points_y, mitosis_height, mitosis_width):
    
    # need to add saving path to save all the patche to train the system in the future... or check whether it is cropping 
    # appropiate patche...
    num_patches = len(cntr_points_x)   
    
    patches_for_test = np.ndarray((num_patches, mitosis_height, mitosis_width,3), dtype=np.uint8)          
    k = 0
    for column,row in zip(cntr_points_x,cntr_points_y):                
            #args.crop_height, args.crop_width            
            #if (row > args.crop_height/2 and column > args.crop_width/2 and row < args.crop_height - args.crop_height/2 and column < args.crop_width - args.crop_width/2):                          
        xmin = int(row - mitosis_height/2)
        ymin = int(column - mitosis_width/2)
        xmax = int(row + mitosis_height/2)
        ymax = int(column + mitosis_width/2)                  
              
        single_patch = img_4_patches[xmin:xmax,ymin:ymax]  
        patches_for_test[k,:,:,:] = single_patch  
  
        # for saving individual patch...         
        #img_name = str(k)+'.jpg'
        #img_path_2save = os.path.join(testing_image_log_saving_path,img_name)    
        #cv2.imwrite(img_path_2save,single_patch)            
                    
        k = k + 1            
    patches_for_test = np.array(patches_for_test).astype(np.float32)

    return patches_for_test 
    

def extract_patches_for_classification_cvpr_2019(img_4_patches, cntr_points_x,cntr_points_y, mitosis_height, mitosis_width):
    
    # need to add saving path to save all the patche to train the system in the future... or check whether it is cropping 
    # appropiate patche...
    num_patches = len(cntr_points_x)   
    
    patches_for_test = np.ndarray((num_patches, mitosis_height, mitosis_width), dtype=np.uint16)          
    k = 0
    for column,row in zip(cntr_points_x,cntr_points_y):                
        xmin = int(row - mitosis_height/2)
        ymin = int(column - mitosis_width/2)
        xmax = int(row + mitosis_height/2)
        ymax = int(column + mitosis_width/2)                  
              
        single_patch = img_4_patches[xmin:xmax,ymin:ymax]  
        patches_for_test[k,:,:] = single_patch  
        k = k + 1            
    patches_for_test = np.array(patches_for_test).astype(np.float32)

    return patches_for_test     
    
def check_xy_center_coordinate (image, x, y,patch_h,patch_w):
    
    height, width, channels = image.shape    
    if x <int(patch_w/2):
        x = int(patch_w/2)
    if y <int(patch_h/2):
        y =int(patch_h/2)    
    if x > int(width - int(patch_w/2)):
        x = int(width - int(patch_w/2))         
    if y > int(height - int(patch_h/2)):
        y = int(height - int(patch_h/2))
    return y, x
    
def extract_patches_from_contour_center(full_img, cntr_points_x,cntr_points_y, patch_h,patch_w):
               
    height,width,channels = full_img.shape        
    #class_name = img_saving_dir.split('/')[-2]    
    #print('class name:'+class_name)    
    pn = 0    
    for x,y in zip(cntr_points_x,cntr_points_y):                
        
        row = int(y)
        column =int(x)
        
        if (row > patch_h/2 and column > patch_w/2 and row < height - patch_h/2 and column < width - patch_w/2):  
            
            row, column = check_xy_center_coordinate (full_img, column, row,patch_h,patch_w)
             
            patch_img = full_img[row-int(patch_h/2):row+int(patch_h/2),column-int(patch_w/2):column+int(patch_w/2),:]           
            img_name_wo_ext =str(img_name)+'_'+str(pn)                  
            #if class_name =='1':
            img_name_w_ext =img_name_wo_ext+'.jpg' 
            final_des_img = os.path.join(img_saving_dir,img_name_w_ext)  
            cv2.imwrite(final_des_img,patch_img) 
                #pdb.set_trace()
            #data_aug_and_save_for_classification(patch_img, img_name_wo_ext, mitosis_agu_per_sample,img_saving_dir)                
            
            '''
            else:
                img_name_w_ext =img_name_wo_ext+'.jpg' 
                final_des_img = os.path.join(img_saving_dir,img_name_w_ext)  
                cv2.imwrite(final_des_img,patch_img)  
            '''      
            pn+=1                
            print ('Processing for: ' +str(pn))    
    return pn
    
def check_coordinates(xmin,ymin,xmax,ymax,mitosis_height, mitosis_width,height, width):
    
    if xmin<0:
        xmin = 0
        xmax = mitosis_width
        
    if ymin<0:
        ymin = 0
        ymax = mitosis_height
    
    if xmax > width:
        xmin = width - mitosis_width
        xmax = width
        
    if ymax > height:
        ymin = height - mitosis_height
        ymax = height
    
    return xmin,ymin,xmax,ymax
        
def check_coordinates_from_centerxy(xmin,ymin,xmax,ymax,mitosis_height, mitosis_width,height, width):
    
    if xmin<0:
        xmin = 0
        xmax = mitosis_width
        
    if ymin<0:
        ymin = 0
        ymax = mitosis_height
    
    if xmax > width:
        xmin = width - mitosis_width
        xmax = width
        
    if ymax > height:
        ymin = height - mitosis_height
        ymax = height
    
    return xmin,ymin,xmax,ymax
    

def extract_9patches_from_image(img_4_patches,cntr_points_x,cntr_points_y, mitosis_height, mitosis_width,stride_pixels):
    
    # extract shape for image and make an array to store image...
    height, width, channels = img_4_patches.shape
    patches_9 = np.ndarray((9, mitosis_height, mitosis_width,3), dtype=np.uint8) 
    
    column = cntr_points_x
    row = cntr_points_y
    # calculate (xmin,ymin,xmax,ymax) for the patch and extract 0 patch
    xmin0 = int(row - mitosis_height/2)
    ymin0 = int(column - mitosis_width/2)
    xmax0 = int(row + mitosis_height/2)
    ymax0 = int(column + mitosis_width/2)
       
    patches_9[0,:,:,:] = img_4_patches[xmin0:xmax0,ymin0:ymax0] 
    
    xmin1 = xmin0-stride_pixels
    ymin1 = ymin0-stride_pixels
    xmax1 = xmax0-stride_pixels
    ymax1 = ymax0-stride_pixels    
    xmin1,ymin1,xmax1,ymax1 = check_coordinates(xmin1,ymin1,xmax1,ymax1,mitosis_height, mitosis_width,height, width)
    patches_9[1,:,:,:] = img_4_patches[xmin1:xmax1,ymin1:ymax1] 
    
    xmin2 = xmin0
    ymin2 = ymin0-stride_pixels
    xmax2 = xmax0
    ymax2 = ymax0-stride_pixels
    xmin2,ymin2,xmax2,ymax2 = check_coordinates(xmin2,ymin2,xmax2,ymax2,mitosis_height, mitosis_width,height, width)
    patches_9[2,:,:,:] = img_4_patches[xmin2:xmax2,ymin2:ymax2] 
        
    xmin3 = xmin0+stride_pixels
    ymin3 = ymin0-stride_pixels
    xmax3 = xmax0+stride_pixels
    ymax3 = ymax0-stride_pixels   
    xmin3,ymin3,xmax3,ymax3 = check_coordinates(xmin3,ymin3,xmax3,ymax3,mitosis_height, mitosis_width,height, width)
    patches_9[3,:,:,:] = img_4_patches[xmin3:xmax3,ymin3:ymax3] 
   
    xmin4 = xmin0+stride_pixels
    ymin4 = ymin0
    xmax4 = xmax0+stride_pixels
    ymax4 = ymax0   
    xmin4,ymin4,xmax4,ymax4 = check_coordinates(xmin4,ymin4,xmax4,ymax4,mitosis_height, mitosis_width,height, width)
    patches_9[4,:,:,:] = img_4_patches[xmin4:xmax4,ymin4:ymax4] 
    
    xmin5 = xmin0+stride_pixels
    ymin5 = ymin0+stride_pixels
    xmax5 = xmax0+stride_pixels
    ymax5 = ymax0+stride_pixels   
    xmin5,ymin5,xmax5,ymax5 = check_coordinates(xmin5,ymin5,xmax5,ymax5,mitosis_height, mitosis_width,height, width)
    patches_9[5,:,:,:] = img_4_patches[xmin5:xmax5,ymin5:ymax5] 
    
    xmin6 = xmin0
    ymin6 = ymin0+stride_pixels
    xmax6 = xmax0
    ymax6 = ymax0+stride_pixels   
    xmin6,ymin6,xmax6,ymax6 = check_coordinates(xmin6,ymin6,xmax6,ymax6,mitosis_height, mitosis_width,height, width)
    patches_9[6,:,:,:] = img_4_patches[xmin6:xmax6,ymin6:ymax6] 
    
    xmin7 = xmin0-stride_pixels
    ymin7 = ymin0+stride_pixels
    xmax7 = xmax0-stride_pixels
    ymax7 = ymax0+stride_pixels   
    xmin7,ymin7,xmax7,ymax7 = check_coordinates(xmin7,ymin7,xmax7,ymax7,mitosis_height, mitosis_width,height, width)
    patches_9[7,:,:,:] = img_4_patches[xmin7:xmax7,ymin7:ymax7] 
    
    xmin8 = xmin0-stride_pixels
    ymin8 = ymin0
    xmax8 = xmax0-stride_pixels
    ymax8 = ymax0   
    xmin8,ymin8,xmax8,ymax8 = check_coordinates(xmin8,ymin8,xmax8,ymax8,mitosis_height, mitosis_width,height, width)
    patches_9[8,:,:,:] = img_4_patches[xmin8:xmax8,ymin8:ymax8]
    
    return patches_9 

def extract_9patches_for_classification(img_4_patches, cntr_points_x,cntr_points_y, mitosis_height, mitosis_width,stride_pixels):
    
    # need to add saving path to save all the patche to train the system in the future... or check whether it is cropping 
    # appropiate patche...
    num_patches = len(cntr_points_x)   
    
    patches_for_test = np.ndarray((9*num_patches, mitosis_height, mitosis_width,3), dtype=np.uint8)          
    k = 0
    num_of_patches = 9
    for column,row in zip(cntr_points_x,cntr_points_y):                
            #args.crop_height, args.crop_width            
            #if (row > args.crop_height/2 and column > args.crop_width/2 and row < args.crop_height - args.crop_height/2 and column < args.crop_width - args.crop_width/2):                          
                 
        patches_9 = extract_9patches_from_image(img_4_patches,column,row, mitosis_height, mitosis_width,stride_pixels)      
        #single_patch = img_4_patches[xmin:xmax,ymin:ymax]  
        patches_for_test[(k*num_of_patches):(k*num_of_patches + num_of_patches),:,:,:] = patches_9  
  
        # for saving individual patch...         
        #img_name = str(k)+'.jpg'
        #img_path_2save = os.path.join(testing_image_log_saving_path,img_name)    
        #cv2.imwrite(img_path_2save,single_patch)            
                    
        k = k + 1            
    patches_for_test = np.array(patches_for_test).astype(np.float32)

    return patches_for_test 


def extract_9patches_from_image_2D(img_4_patches,cntr_points_x,cntr_points_y, mitosis_height, mitosis_width,stride_pixels):
    
    # extract shape for image and make an array to store image...
    height, width = img_4_patches.shape
    patches_9 = np.ndarray((9, mitosis_height, mitosis_width), dtype=np.uint8) 
    
    column = cntr_points_x
    row = cntr_points_y
    # calculate (xmin,ymin,xmax,ymax) for the patch and extract 0 patch
    xmin0 = int(row - mitosis_height/2)
    ymin0 = int(column - mitosis_width/2)
    xmax0 = int(row + mitosis_height/2)
    ymax0 = int(column + mitosis_width/2)
       
    patches_9[0,:,:] = cv2.resize(img_4_patches[xmin0:xmax0,ymin0:ymax0], dsize=(mitosis_height, mitosis_width), interpolation=cv2.INTER_NEAREST) 
        
    xmin1 = xmin0-stride_pixels
    ymin1 = ymin0-stride_pixels
    xmax1 = xmax0-stride_pixels
    ymax1 = ymax0-stride_pixels    
    xmin1,ymin1,xmax1,ymax1 = check_coordinates(xmin1,ymin1,xmax1,ymax1,mitosis_height, mitosis_width,height, width)
    patches_9[1,:,:] = cv2.resize(img_4_patches[xmin1:xmax1,ymin1:ymax1], dsize=(mitosis_height, mitosis_width), interpolation=cv2.INTER_NEAREST)  
    
    xmin2 = xmin0
    ymin2 = ymin0-stride_pixels
    xmax2 = xmax0
    ymax2 = ymax0-stride_pixels
    xmin2,ymin2,xmax2,ymax2 = check_coordinates(xmin2,ymin2,xmax2,ymax2,mitosis_height, mitosis_width,height, width)
    patches_9[2,:,:] = cv2.resize(img_4_patches[xmin2:xmax2,ymin2:ymax2], dsize=(mitosis_height, mitosis_width), interpolation=cv2.INTER_NEAREST)  
        
    xmin3 = xmin0+stride_pixels
    ymin3 = ymin0-stride_pixels
    xmax3 = xmax0+stride_pixels
    ymax3 = ymax0-stride_pixels   
    xmin3,ymin3,xmax3,ymax3 = check_coordinates(xmin3,ymin3,xmax3,ymax3,mitosis_height, mitosis_width,height, width)
    patches_9[3,:,:] = cv2.resize(img_4_patches[xmin3:xmax3,ymin3:ymax3], dsize=(mitosis_height, mitosis_width), interpolation=cv2.INTER_NEAREST)  
   
    xmin4 = xmin0+stride_pixels
    ymin4 = ymin0
    xmax4 = xmax0+stride_pixels
    ymax4 = ymax0   
    xmin4,ymin4,xmax4,ymax4 = check_coordinates(xmin4,ymin4,xmax4,ymax4,mitosis_height, mitosis_width,height, width)
    patches_9[4,:,:] = cv2.resize(img_4_patches[xmin4:xmax4,ymin4:ymax4], dsize=(mitosis_height, mitosis_width), interpolation=cv2.INTER_NEAREST)  
    
    xmin5 = xmin0+stride_pixels
    ymin5 = ymin0+stride_pixels
    xmax5 = xmax0+stride_pixels
    ymax5 = ymax0+stride_pixels   
    xmin5,ymin5,xmax5,ymax5 = check_coordinates(xmin5,ymin5,xmax5,ymax5,mitosis_height, mitosis_width,height, width)
    patches_9[5,:,:] = cv2.resize(img_4_patches[xmin5:xmax5,ymin5:ymax5], dsize=(mitosis_height, mitosis_width), interpolation=cv2.INTER_NEAREST)  

    xmin6 = xmin0
    ymin6 = ymin0+stride_pixels
    xmax6 = xmax0
    ymax6 = ymax0+stride_pixels   
    xmin6,ymin6,xmax6,ymax6 = check_coordinates(xmin6,ymin6,xmax6,ymax6,mitosis_height, mitosis_width,height, width)
    patches_9[6,:,:] = cv2.resize(img_4_patches[xmin6:xmax6,ymin6:ymax6], dsize=(mitosis_height, mitosis_width), interpolation=cv2.INTER_NEAREST)  
    
    xmin7 = xmin0-stride_pixels
    ymin7 = ymin0+stride_pixels
    xmax7 = xmax0-stride_pixels
    ymax7 = ymax0+stride_pixels   
    xmin7,ymin7,xmax7,ymax7 = check_coordinates(xmin7,ymin7,xmax7,ymax7,mitosis_height, mitosis_width,height, width)
    patches_9[7,:,:] = cv2.resize(img_4_patches[xmin7:xmax7,ymin7:ymax7], dsize=(mitosis_height, mitosis_width), interpolation=cv2.INTER_NEAREST)  
    
    xmin8 = xmin0-stride_pixels
    ymin8 = ymin0
    xmax8 = xmax0-stride_pixels
    ymax8 = ymax0   
    xmin8,ymin8,xmax8,ymax8 = check_coordinates(xmin8,ymin8,xmax8,ymax8,mitosis_height, mitosis_width,height, width)
    patches_9[8,:,:] = cv2.resize(img_4_patches[xmin8:xmax8,ymin8:ymax8], dsize=(mitosis_height, mitosis_width), interpolation=cv2.INTER_NEAREST) 
    
    return patches_9 

def extract_9patches_for_classification_2D(img_4_patches, cntr_points_x,cntr_points_y, mitosis_height, mitosis_width,stride_pixels):
    
    # need to add saving path to save all the patche to train the system in the future... or check whether it is cropping 
    # appropiate patche...
    num_patches = len(cntr_points_x)   
    
    patches_for_test = np.ndarray((9*num_patches, mitosis_height, mitosis_width), dtype=np.uint16)          
    k = 0
    num_of_patches = 9
    for column,row in zip(cntr_points_x,cntr_points_y):                
            #args.crop_height, args.crop_width            
            #if (row > args.crop_height/2 and column > args.crop_width/2 and row < args.crop_height - args.crop_height/2 and column < args.crop_width - args.crop_width/2):                          

        patches_9 = extract_9patches_from_image_2D(img_4_patches,column,row, mitosis_height, mitosis_width,stride_pixels)      
        #single_patch = img_4_patches[xmin:xmax,ymin:ymax]  
        patches_for_test[(k*num_of_patches):(k*num_of_patches + num_of_patches),:,:] = patches_9  
  
        # for saving individual patch...         
        #img_name = str(k)+'.jpg'
        #img_path_2save = os.path.join(testing_image_log_saving_path,img_name)    
        #cv2.imwrite(img_path_2save,single_patch)            
                    
        k = k + 1            
    patches_for_test = np.array(patches_for_test).astype(np.float32)

    return patches_for_test 
 
def extract_9patches_for_classification_2D_cvpr_2019(img_4_patches, cntr_points_x,cntr_points_y, mitosis_height, mitosis_width,stride_pixels):    
    # need to add saving path to save all the patche to train the system in the future... or check whether it is cropping 
    # appropiate patche...
    num_patches = len(cntr_points_x)       
    patches_for_test = np.ndarray((num_patches, mitosis_height, mitosis_width), dtype=np.uint8)          
   
   
    num_of_patches = 9
    k = 0
    for column,row in zip(cntr_points_x,cntr_points_y):                
        xmin0 = int(row - mitosis_height/2)
        ymin0 = int(column - mitosis_width/2)
        xmax0 = int(row + mitosis_height/2)
        ymax0 = int(column + mitosis_width/2)       
        patches_for_test[k,:,:] = cv2.resize(img_4_patches[xmin0:xmax0,ymin0:ymax0], dsize=(mitosis_height, mitosis_width), interpolation=cv2.INTER_NEAREST)           
        k = k + 1  
          
    patches_for_test = np.array(patches_for_test).astype(np.float32)

    return patches_for_test 
    
def conf_vec_to_matrix_to_vec_for_9patches(conf_vec):
    
    length = len(conf_vec)
    matrix_length = int(length/9)    
    conf_matrix = np.zeros((matrix_length,9),dtype=float)  
    num_of_patches = 9
    for i in range(matrix_length):
        conf_matrix[i,:] = conf_vec[i*num_of_patches:(i*num_of_patches+num_of_patches)]    
    # select maximum values per row...finral conf. vector..    
    new_confv_vec = conf_matrix.max(1)      
    # calculate the mean confident vector...
    #new_confv_vec = conf_matrix.mean(1)
     
    return new_confv_vec
    
def extract_patches_binary_mask_for_classification(img_4_masks, cntr_points_x,cntr_points_y, mitosis_height, mitosis_width):
    
    # need to add saving path to save all the patche to train the system in the future... or check whether it is cropping 
    # appropiate patche...
    num_patches = len(cntr_points_x)   
    
    patches_masks = np.ndarray((num_patches, mitosis_height, mitosis_width), dtype=np.uint8)          
    k = 0
    for column,row in zip(cntr_points_x,cntr_points_y):                
            #args.crop_height, args.crop_width            
            #if (row > args.crop_height/2 and column > args.crop_width/2 and row < args.crop_height - args.crop_height/2 and column < args.crop_width - args.crop_width/2):                          
        xmin = int(row - mitosis_height/2)
        ymin = int(column - mitosis_width/2)
        xmax = int(row + mitosis_height/2)
        ymax = int(column + mitosis_width/2)                  
              
        single_patch_mask = img_4_masks[xmin:xmax,ymin:ymax]  
        patches_masks[k,:,:] = single_patch_mask  
  
        # for saving individual patch...         
        #img_name = str(k)+'.jpg'
        #img_path_2save = os.path.join(testing_image_log_saving_path,img_name)    
        #cv2.imwrite(img_path_2save,single_patch)            
                    
        k = k + 1            
    patches_masks = np.array(patches_masks).astype(np.float32)

    return patches_masks   
    

def image_with_confidence(img_4_conf,cntr_points_x,cntr_points_y,conf_vec,threshold):
    
    m_points_x = []
    m_points_y = [] 
    m_confident = []
    total_mitosis = 0
    for x,y,c in zip(cntr_points_x,cntr_points_y,conf_vec):  
            #print(str(c))
        if c>threshold:
            
            m_points_x.append(x)
            m_points_y.append(y)
            m_confident.append(c)
            total_mitosis +=1
            cv2.putText(img_4_conf,str(c),(int(x),int(y)), font,0.3,(0,0,0))        
            cv2.rectangle(img_4_conf,(x-15,y-15),(x+15,y+15),(0,255,0),2)
            #cv2.circle(img_4_conf, (x, y), 2, (0,0,255), -1)
            
    cv2.putText(img_4_conf,'Total : '+str(total_mitosis),(int(30),int(30)), font,1,(255,0,0))     
    
    return img_4_conf,total_mitosis,m_points_x, m_points_y,m_confident

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
    
def foregournd_pixel_image_from_mask(image,mask):
    
    height, width, channels = image.shape   
    foreground_image = np.zeros((height, width, channels),dtype=np.uint8)
    
    mask = mask > 125
    for row in range(height):
      for column in range(width):
          
         pixel_values = mask[row,column]
         
         if pixel_values == 1:
             foreground_image[row,column,:] = image[row,column,:]
         else:
             foreground_image[row,column,:] = [0,0,0]

    return foreground_image

def foregournd_pixel_image_from_mask_cvpr_2019(image,mask):
    
    height, width = image.shape   
    foreground_image = np.zeros((height, width),dtype=np.uint16)
    
    mask = mask > 125
    for row in range(height):
      for column in range(width):
          
         pixel_values = mask[row,column]
         
         if pixel_values == 1:
             foreground_image[row,column] = image[row,column]
         else:
             foreground_image[row,column] = 0

    return foreground_image
    
def draw_rectangle_for_gt(img_4_conf,cntr_points_x,cntr_points_y):
    
    m_points_x = []
    m_points_y = [] 
    #m_confident = []
    total_mitosis = 0
    
    for x,y in zip(cntr_points_x,cntr_points_y):             
        m_points_x.append(x)
        m_points_y.append(y)
        total_mitosis +=1
        #cv2.putText(img_4_conf,str(c),(int(x),int(y)), font,0.3,(0,255,0))        
        cv2.rectangle(img_4_conf,(x-15,y-15),(x+15,y+15),(255,0,0),2)
        #cv2.circle(img_4_conf, (x, y), 2, (0,0,255), -1)
            
    #cv2.putText(img_4_conf,'Total : '+str(total_mitosis),(int(30),int(30)), font,1,(0,255,0))     
    
    return img_4_conf
    
def extract_contour_centers(pred_mask,mitosis_width,mitosis_height):
         
    #pred_reverse=255-pred  
    #pred_reverse=pred_reverse.astype('uint8')    
    
    cntr_points_x = []
    cntr_points_y = []    
    # Save the (x,y) coordinate of center pixel... if the coordinate is in the border then apply condition as follows:   
    #pdb.set_trace() 
    #out_img = original_img
    temp_bin_img = pred_mask.copy()    
    #ret, thresh = cv2.threshold(temp_bin_img, 127, 255, 0)
    temp_bin_img = temp_bin_img.astype('uint8')
    #thresh =thresh.astype('uint8') 
    im2, contours, hierarchy = cv2.findContours(temp_bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
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

def fill_true_positive_from_pred_mask(pred_mask,gt_contour_cx,gt_contour_cy, patch_w, patch_h):
    
    number_of_points = len(gt_contour_cx)
    
    height, width = pred_mask.shape
    
    for x,y in zip(gt_contour_cx,gt_contour_cy):
        
        m_row = int(y)
        m_column = int(x)           
        #if m_row > 2*patch_h or m_column > 2*patch_w or 

        if m_row+int(patch_h*2) > height:
            h_direc_higher = int(abs(height-m_row))
        else:
            h_direc_higher = m_row+int(patch_h*2)
            
        if m_row-int(patch_h*2) < 0:
            h_direc_lower = 0
        else:
            h_direc_lower = m_row-int(patch_h*2)
            
        if m_column+int(patch_w*2) > width:
            w_direc_higher = int(abs(width-m_column))
        else:
            w_direc_higher = m_column+int(width*2)
        
        if m_column-int(patch_w*2) < 0:
            w_direc_lower = 0
        else:
            w_direc_lower = m_column-int(patch_w*2)
             
        #pred_mask[m_row-int(patch_h*2):m_row+int(patch_h*2),m_column-int(patch_w*2):m_column+int(patch_w*2)] = 0
        
        pred_mask[h_direc_lower:h_direc_higher,w_direc_lower:w_direc_higher] = 0
        #print(cordinate)
    
    fp_pred_mask=pred_mask
    
    return fp_pred_mask
    
def save_patches(patches, img_name, image_saving_dir):
    
    num_image,image_h,image_w,channels = patches.shape
    
    for i in range(num_image):
        name = img_name+'_'+str(i)+'.jpg'
        final_saving_path = join_path(image_saving_dir,name)
        patch = np.squeeze(patches[i,:,:,:])        
        cv2.imwrite(final_saving_path,patch)
    
    return 0

def save_patches_2D_cvpr_2019(patches, img_name, image_saving_dir):
    
    num_image,image_h,image_w = patches.shape
    
    for i in range(num_image):
        name = img_name+'_'+str(i)+'.tif'
        final_saving_path = join_path(image_saving_dir,name)
        patch = np.squeeze(patches[i,:,:])        
        cv2.imwrite(final_saving_path,patch)
    
    return 0
        
def save_masks(patches, img_name, image_saving_dir):
    
    num_image,image_h,image_w = patches.shape
    
    for i in range(num_image):
        name = img_name+'_'+str(i)+'.jpg'
        final_saving_path = join_path(image_saving_dir,name)
        patch = np.squeeze(patches[i,:,:])        
        cv2.imwrite(final_saving_path,patch)
    
    return 0
        

def save_false_true_positive_patches_from_prediction_mask(image,pred_m_point_x,pred_m_point_y,gt_contour_cx,gt_contour_cy,mpatch_w, mpatch_h,img_name,true_false_positive_saving_path):
    
    if not os.path.isdir("%s/%s"%(true_false_positive_saving_path,'mitosis')):
        os.makedirs("%s/%s"%(true_false_positive_saving_path,'mitosis'))
    
    if not os.path.isdir("%s/%s"%(true_false_positive_saving_path,'non_mitosis')):
        os.makedirs("%s/%s"%(true_false_positive_saving_path,'non_mitosis'))

    final_true_positive_saving_path_mitosis = join_path(true_false_positive_saving_path,'mitosis/')
    final_false_positive_saving_path_non_mitosis = join_path(true_false_positive_saving_path,'non_mitosis/')

    if not os.path.isdir("%s/%s"%(true_false_positive_saving_path,'mitosis_mask')):
        os.makedirs("%s/%s"%(true_false_positive_saving_path,'mitosis_mask'))
    
    if not os.path.isdir("%s/%s"%(true_false_positive_saving_path,'non_mitosis_mask')):
        os.makedirs("%s/%s"%(true_false_positive_saving_path,'non_mitosis_mask'))

    final_true_positive_saving_path_mitosis_mask = join_path(true_false_positive_saving_path,'mitosis_mask/')
    final_false_positive_saving_path_non_mitosis_mask = join_path(true_false_positive_saving_path,'non_mitosis_mask/')
    
    image_h,image_w,channels = image.shape
    
    mask_initial = np.zeros((image_h,image_w), dtype='float32')
    
    print('Number detected region:'+str(len(pred_m_point_x)))
    
    for row,column in zip(pred_m_point_y,pred_m_point_x):
        mask_initial[row,column] = 255
    
    # create gaussian mask...
    mask_initial = 255.0*(mask_initial[:,:]> 0)
    mask_initial = cv2.dilate(mask_initial,kernel,iterations = 1)
    mask_initial = ndimage.gaussian_filter(mask_initial, sigma=(1,1),order = 0)     
    mask_initial = 255.0*(mask_initial[:,:]> 0.3)
        
    fp_pred_mask = fill_true_positive_from_pred_mask(mask_initial,gt_contour_cx,gt_contour_cy,mpatch_w, mpatch_h)
    

    # test_saving....
    test_img_name = 'first_level_predicted_mask.jpg'
    img_saving_path = join_path(true_false_positive_saving_path,test_img_name)   
    cv2.imwrite(img_saving_path,mask_initial)
    
        
    second_level_pred_contour_cx,second_level_pred_contour_cy = extract_contour_centers(fp_pred_mask,mpatch_w, mpatch_h)
    
    mitosis_patches = extract_patches_for_classification(image, gt_contour_cx,gt_contour_cy, mpatch_w, mpatch_h)
    
    non_mitosis_patches = extract_patches_for_classification(image, second_level_pred_contour_cx,second_level_pred_contour_cy, mpatch_w, mpatch_h)
    
    #pdb.set_trace()
    # save mitosis_region..
    save_patches(mitosis_patches,img_name,final_true_positive_saving_path_mitosis)    
    # save non_mitosis region here...
    save_patches(non_mitosis_patches,img_name,final_false_positive_saving_path_non_mitosis)
    
    return 0



def save_false_true_positive_patches_from_prediction_mask_v2(image,image_mask,pred_m_point_x,pred_m_point_y,gt_contour_cx,gt_contour_cy,mpatch_w, mpatch_h,image_name,true_false_positive_saving_path):
    
    if not os.path.isdir("%s/%s"%(true_false_positive_saving_path,'mitosis')):
        os.makedirs("%s/%s"%(true_false_positive_saving_path,'mitosis'))
    
    if not os.path.isdir("%s/%s"%(true_false_positive_saving_path,'non_mitosis')):
        os.makedirs("%s/%s"%(true_false_positive_saving_path,'non_mitosis'))

    final_true_positive_saving_path_mitosis = join_path(true_false_positive_saving_path,'mitosis/')
    final_false_positive_saving_path_non_mitosis = join_path(true_false_positive_saving_path,'non_mitosis/')

    if not os.path.isdir("%s/%s"%(true_false_positive_saving_path,'mitosis_rgb_mask')):
        os.makedirs("%s/%s"%(true_false_positive_saving_path,'mitosis_rgb_mask'))
    
    if not os.path.isdir("%s/%s"%(true_false_positive_saving_path,'non_mitosis_rgb_mask')):
        os.makedirs("%s/%s"%(true_false_positive_saving_path,'non_mitosis_rgb_mask'))

    final_true_positive_saving_path_mitosis_rgb_mask = join_path(true_false_positive_saving_path,'mitosis_rgb_mask/')
    final_false_positive_saving_path_non_mitosis_rgb_mask = join_path(true_false_positive_saving_path,'non_mitosis_rgb_mask/')
    
    if not os.path.isdir("%s/%s"%(true_false_positive_saving_path,'mitosis_mask')):
        os.makedirs("%s/%s"%(true_false_positive_saving_path,'mitosis_mask'))
    
    if not os.path.isdir("%s/%s"%(true_false_positive_saving_path,'non_mitosis_mask')):
        os.makedirs("%s/%s"%(true_false_positive_saving_path,'non_mitosis_mask'))

    final_true_positive_saving_path_mitosis_mask = join_path(true_false_positive_saving_path,'mitosis_mask/')
    final_false_positive_saving_path_non_mitosis_mask = join_path(true_false_positive_saving_path,'non_mitosis_mask/')
    
    image_h,image_w,channels = image.shape
    
    mask_initial = np.zeros((image_h,image_w), dtype='float32')
    
    print('Number detected region:'+str(len(pred_m_point_x)))
    
    for row,column in zip(pred_m_point_y,pred_m_point_x):
        mask_initial[row,column] = 255
    
    # create gaussian mask...
    mask_initial = 255.0*(mask_initial[:,:]> 0)
    mask_initial = cv2.dilate(mask_initial,kernel,iterations = 1)
    mask_initial = ndimage.gaussian_filter(mask_initial, sigma=(1,1),order = 0)     
    mask_initial = 255.0*(mask_initial[:,:]> 0.3)
        
    fp_pred_mask = fill_true_positive_from_pred_mask(mask_initial,gt_contour_cx,gt_contour_cy,mpatch_w, mpatch_h)
    

    # test_saving....
    #test_img_name = 'first_level_predicted_mask.jpg'
    #img_saving_path = join_path(true_false_positive_saving_path,test_img_name)   
    #cv2.imwrite(img_saving_path,mask_initial)
    
        
    second_level_pred_contour_cx,second_level_pred_contour_cy = extract_contour_centers(fp_pred_mask,mpatch_w, mpatch_h)
    
    # forground image extraction from input image.....
    foregournd_image = foregournd_pixel_image_from_mask(image,image_mask) 
    
    # Extract the image pathces...
    mitosis_patches = extract_patches_for_classification(image, gt_contour_cx,gt_contour_cy, mpatch_w, mpatch_h)
    non_mitosis_patches = extract_patches_for_classification(image, second_level_pred_contour_cx,second_level_pred_contour_cy, mpatch_w, mpatch_h)
    
    # Extract RGB mask ... from image..    
    mitosis_rgb_mask = extract_patches_for_classification(foregournd_image, gt_contour_cx,gt_contour_cy, mpatch_w, mpatch_h)
    non_mitosis_rgb_mask = extract_patches_for_classification(foregournd_image, second_level_pred_contour_cx,second_level_pred_contour_cy, mpatch_w, mpatch_h)
    
    # Extract the image pathces masks...
    mitosis_patches_masks = extract_patches_binary_mask_for_classification(image_mask, gt_contour_cx,gt_contour_cy, mpatch_w, mpatch_h)
    non_mitosis_patches_mask = extract_patches_binary_mask_for_classification(image_mask, second_level_pred_contour_cx,second_level_pred_contour_cy, mpatch_w, mpatch_h)

    
    #pdb.set_trace()
    # save mitosis_region and non-mitosis region pathces.....
    save_patches(mitosis_patches,image_name,final_true_positive_saving_path_mitosis) 
    save_patches(non_mitosis_patches,image_name,final_false_positive_saving_path_non_mitosis)
    
    # save mitosis_region and non-mitosis region pathces.....
    rgb_mask_name = image_name+'_rgb_mask'
    save_patches(mitosis_rgb_mask,rgb_mask_name,final_true_positive_saving_path_mitosis_rgb_mask) 
    save_patches(non_mitosis_rgb_mask,rgb_mask_name,final_false_positive_saving_path_non_mitosis_rgb_mask)
    
    # save the mask for mitosis and non_mitosis region here...
    mask_name = image_name+'_mask'
    save_masks(mitosis_patches_masks,mask_name,final_true_positive_saving_path_mitosis_mask) 
    save_masks(non_mitosis_patches_mask,mask_name,final_false_positive_saving_path_non_mitosis_mask)
    
    return 0

def save_false_true_positive_patches_from_prediction_mask_cvpr2019(image,image_mask,pred_m_point_x,pred_m_point_y,gt_contour_cx,gt_contour_cy,mpatch_w, mpatch_h,image_name,true_false_positive_saving_path):
    
    if not os.path.isdir("%s/%s"%(true_false_positive_saving_path,'mitosis')):
        os.makedirs("%s/%s"%(true_false_positive_saving_path,'mitosis'))
    
    if not os.path.isdir("%s/%s"%(true_false_positive_saving_path,'non_mitosis')):
        os.makedirs("%s/%s"%(true_false_positive_saving_path,'non_mitosis'))

    final_true_positive_saving_path_mitosis = join_path(true_false_positive_saving_path,'mitosis/')
    final_false_positive_saving_path_non_mitosis = join_path(true_false_positive_saving_path,'non_mitosis/')

    if not os.path.isdir("%s/%s"%(true_false_positive_saving_path,'mitosis_rgb_mask')):
        os.makedirs("%s/%s"%(true_false_positive_saving_path,'mitosis_rgb_mask'))
    
    if not os.path.isdir("%s/%s"%(true_false_positive_saving_path,'non_mitosis_rgb_mask')):
        os.makedirs("%s/%s"%(true_false_positive_saving_path,'non_mitosis_rgb_mask'))

    final_true_positive_saving_path_mitosis_rgb_mask = join_path(true_false_positive_saving_path,'mitosis_rgb_mask/')
    final_false_positive_saving_path_non_mitosis_rgb_mask = join_path(true_false_positive_saving_path,'non_mitosis_rgb_mask/')
    
    if not os.path.isdir("%s/%s"%(true_false_positive_saving_path,'mitosis_mask')):
        os.makedirs("%s/%s"%(true_false_positive_saving_path,'mitosis_mask'))
    
    if not os.path.isdir("%s/%s"%(true_false_positive_saving_path,'non_mitosis_mask')):
        os.makedirs("%s/%s"%(true_false_positive_saving_path,'non_mitosis_mask'))

    final_true_positive_saving_path_mitosis_mask = join_path(true_false_positive_saving_path,'mitosis_mask/')
    final_false_positive_saving_path_non_mitosis_mask = join_path(true_false_positive_saving_path,'non_mitosis_mask/')
    
    image_h,image_w = image.shape
    
    mask_initial = np.zeros((image_h,image_w), dtype='float32')
    
    print('Number detected region:'+str(len(pred_m_point_x)))
    
    for row,column in zip(pred_m_point_y,pred_m_point_x):
        mask_initial[row,column] = 255
    
    # create gaussian mask...
    mask_initial = 255.0*(mask_initial[:,:]> 0)
    mask_initial = cv2.dilate(mask_initial,kernel,iterations = 1)
    mask_initial = ndimage.gaussian_filter(mask_initial, sigma=(1,1),order = 0)     
    mask_initial = 255.0*(mask_initial[:,:]> 0.3)
        
    fp_pred_mask = fill_true_positive_from_pred_mask(mask_initial,gt_contour_cx,gt_contour_cy,mpatch_w, mpatch_h)
    

    # test_saving....
    #test_img_name = 'first_level_predicted_mask.jpg'
    #img_saving_path = join_path(true_false_positive_saving_path,test_img_name)   
    #cv2.imwrite(img_saving_path,mask_initial)
    
        
    second_level_pred_contour_cx,second_level_pred_contour_cy = extract_contour_centers(fp_pred_mask,mpatch_w, mpatch_h)
    
    # forground image extraction from input image.....
    foregournd_image = foregournd_pixel_image_from_mask_cvpr_2019(image,image_mask) 
    
    # Extract the image pathces...
    mitosis_patches = extract_patches_for_classification_cvpr_2019(image, gt_contour_cx,gt_contour_cy, mpatch_w, mpatch_h)
    non_mitosis_patches = extract_patches_for_classification_cvpr_2019(image, second_level_pred_contour_cx,second_level_pred_contour_cy, mpatch_w, mpatch_h)
    
    # Extract RGB mask ... from image..    
    mitosis_rgb_mask = extract_patches_for_classification_cvpr_2019(foregournd_image, gt_contour_cx,gt_contour_cy, mpatch_w, mpatch_h)
    non_mitosis_rgb_mask = extract_patches_for_classification_cvpr_2019(foregournd_image, second_level_pred_contour_cx,second_level_pred_contour_cy, mpatch_w, mpatch_h)
    
    # Extract the image pathces masks...
    mitosis_patches_masks = extract_patches_binary_mask_for_classification(image_mask, gt_contour_cx,gt_contour_cy, mpatch_w, mpatch_h)
    non_mitosis_patches_mask = extract_patches_binary_mask_for_classification(image_mask, second_level_pred_contour_cx,second_level_pred_contour_cy, mpatch_w, mpatch_h)

    
    #pdb.set_trace()
    # save mitosis_region and non-mitosis region pathces.....
    #save_patches_2D(mitosis_patches,image_name,final_true_positive_saving_path_mitosis) 
    
    cvpr_2019_data_demo.saving_patches(mitosis_patches,image_name,final_true_positive_saving_path_mitosis) 
    
    cvpr_2019_data_demo.saving_patches(non_mitosis_patches,image_name,final_false_positive_saving_path_non_mitosis)
    
    # save mitosis_region and non-mitosis region pathces.....
    #rgb_mask_name = image_name+'_rgb_mask'
    #save_patches_2D_cvpr_2019(mitosis_rgb_mask,rgb_mask_name,final_true_positive_saving_path_mitosis_rgb_mask) 
    #save_patches_2D_cvpr_2019(non_mitosis_rgb_mask,rgb_mask_name,final_false_positive_saving_path_non_mitosis_rgb_mask)
    
    # save the mask for mitosis and non_mitosis region here...
    #mask_name = image_name+'_mask'
    #save_masks(mitosis_patches_masks,mask_name,final_true_positive_saving_path_mitosis_mask) 
    #save_masks(non_mitosis_patches_mask,mask_name,final_false_positive_saving_path_non_mitosis_mask)
    
    return 0    

def save_false_true_positive_patches_from_prediction_mask_cvpr2019_single_dir(image,image_mask,pred_m_point_x,pred_m_point_y,gt_contour_cx,gt_contour_cy,mpatch_w, mpatch_h,image_name,true_false_positive_saving_path):
    
    
    image_h,image_w = image.shape
    
    mask_initial = np.zeros((image_h,image_w), dtype='float32')
    
    print('Number detected region:'+str(len(pred_m_point_x)))
    
    for row,column in zip(pred_m_point_y,pred_m_point_x):
        mask_initial[row,column] = 255
    
    # create gaussian mask...
    mask_initial = 255.0*(mask_initial[:,:]> 0)
    mask_initial = cv2.dilate(mask_initial,kernel,iterations = 1)
    mask_initial = ndimage.gaussian_filter(mask_initial, sigma=(1,1),order = 0)     
    mask_initial = 255.0*(mask_initial[:,:]> 0.3)
        
    #fp_pred_mask = fill_true_positive_from_pred_mask(mask_initial,gt_contour_cx,gt_contour_cy,mpatch_w, mpatch_h)
    

    # test_saving....
    #test_img_name = 'first_level_predicted_mask.jpg'
    #img_saving_path = join_path(true_false_positive_saving_path,test_img_name)   
    #cv2.imwrite(img_saving_path,mask_initial)
    
        
    #second_level_pred_contour_cx,second_level_pred_contour_cy = extract_contour_centers(fp_pred_mask,mpatch_w, mpatch_h)
    
    # forground image extraction from input image.....
    #foregournd_image = foregournd_pixel_image_from_mask_cvpr_2019(image,image_mask) 
    
    # Extract the image pathces...
    #mitosis_patches = extract_patches_for_classification_cvpr_2019(image, gt_contour_cx,gt_contour_cy, mpatch_w, mpatch_h)
    mitosis_non_mitosis_patches = extract_patches_for_classification_cvpr_2019(image, pred_m_point_x,pred_m_point_y, mpatch_w, mpatch_h)
    
    '''
    # Extract RGB mask ... from image..    
    mitosis_rgb_mask = extract_patches_for_classification_cvpr_2019(foregournd_image, gt_contour_cx,gt_contour_cy, mpatch_w, mpatch_h)
    non_mitosis_rgb_mask = extract_patches_for_classification_cvpr_2019(foregournd_image, second_level_pred_contour_cx,second_level_pred_contour_cy, mpatch_w, mpatch_h)
    
    # Extract the image pathces masks...
    mitosis_patches_masks = extract_patches_binary_mask_for_classification(image_mask, gt_contour_cx,gt_contour_cy, mpatch_w, mpatch_h)
    non_mitosis_patches_mask = extract_patches_binary_mask_for_classification(image_mask, second_level_pred_contour_cx,second_level_pred_contour_cy, mpatch_w, mpatch_h)

    
    #pdb.set_trace()
    # save mitosis_region and non-mitosis region pathces.....
    #save_patches_2D(mitosis_patches,image_name,final_true_positive_saving_path_mitosis) 
    
    cvpr_2019_data_demo.saving_patches(mitosis_patches,image_name,final_true_positive_saving_path_mitosis) 
    '''
    cvpr_2019_data_demo.saving_patches(mitosis_non_mitosis_patches,image_name,final_false_positive_saving_path_non_mitosis)
    
    # save mitosis_region and non-mitosis region pathces.....
    #rgb_mask_name = image_name+'_rgb_mask'
    #save_patches_2D_cvpr_2019(mitosis_rgb_mask,rgb_mask_name,final_true_positive_saving_path_mitosis_rgb_mask) 
    #save_patches_2D_cvpr_2019(non_mitosis_rgb_mask,rgb_mask_name,final_false_positive_saving_path_non_mitosis_rgb_mask)
    
    # save the mask for mitosis and non_mitosis region here...
    #mask_name = image_name+'_mask'
    #save_masks(mitosis_patches_masks,mask_name,final_true_positive_saving_path_mitosis_mask) 
    #save_masks(non_mitosis_patches_mask,mask_name,final_false_positive_saving_path_non_mitosis_mask)
    
    return 0  

'''
def save_false_true_positive_mask_from_entire_prediction_mask(image,pred_m_point_x,pred_m_point_y,gt_contour_cx,gt_contour_cy,mpatch_w, mpatch_h,img_name,true_false_positive_saving_path):
    
    if not os.path.isdir("%s/%s"%(true_false_positive_saving_path,'mitosis')):
        os.makedirs("%s/%s"%(true_false_positive_saving_path,'mitosis'))
    
    if not os.path.isdir("%s/%s"%(true_false_positive_saving_path,'non_mitosis')):
        os.makedirs("%s/%s"%(true_false_positive_saving_path,'non_mitosis'))

    final_true_positive_saving_path_mitosis = join_path(true_false_positive_saving_path,'mitosis/')
    final_false_positive_saving_path_non_mitosis = join_path(true_false_positive_saving_path,'non_mitosis/')

    image_h,image_w = image.shape
    
    mask_initial = np.zeros((image_h,image_w), dtype='float32')
    
    print('Number detected region:'+str(len(pred_m_point_x)))
    
    for row,column in zip(pred_m_point_y,pred_m_point_x):
        mask_initial[row,column] = 255
    
    # create gaussian mask...
    mask_initial = 255.0*(mask_initial[:,:]> 0)
    mask_initial = cv2.dilate(mask_initial,kernel,iterations = 1)
    mask_initial = ndimage.gaussian_filter(mask_initial, sigma=(1,1),order = 0)     
    mask_initial = 255.0*(mask_initial[:,:]> 0.3)
        
    fp_pred_mask = fill_true_positive_from_pred_mask(mask_initial,gt_contour_cx,gt_contour_cy,mpatch_w, mpatch_h)
    

    # test_saving....
    test_img_name = 'first_level_predicted_mask.jpg'
    img_saving_path = join_path(true_false_positive_saving_path,test_img_name)   
    cv2.imwrite(img_saving_path,mask_initial)
    
        
    second_level_pred_contour_cx,second_level_pred_contour_cy = extract_contour_centers(fp_pred_mask,mpatch_w, mpatch_h)
    
    mitosis_patches = extract_patches_for_classification(image, gt_contour_cx,gt_contour_cy, mpatch_w, mpatch_h)
    
    non_mitosis_patches = extract_patches_for_classification(image, second_level_pred_contour_cx,second_level_pred_contour_cy, mpatch_w, mpatch_h)
    
    #pdb.set_trace()
    # save mitosis_region..
    save_patches(mitosis_patches,img_name,final_true_positive_saving_path_mitosis)    
    # save non_mitosis region here...
    save_patches(non_mitosis_patches,img_name,final_false_positive_saving_path_non_mitosis)
    
    return 0
      

def point_clouding_pred_gt(pred, gt, original_img):

    # >>>>>> point clouding for gront truth and predicted images <<<<<<<<<<
    image_h, image_w = pred.shape  
    pd_points_x = []
    pd_points_y = []    
    for i in range(image_h):
       for j in range(image_w):
          if(pred[i,j] == 255):              
              pd_points_x.append(j)
              pd_points_y.append(i)

    #Take the pixels coordinate for predicted region from ground truth.
    image_h, image_w = gt.shape    
    ac_points_x = []
    ac_points_y = []    
    for i in range(image_h):
       for j in range(image_w):
          if(gt[i,j] >= 1 ):             
              ac_points_x.append(j)
              ac_points_y.append(i) 
              
    #img_4_dot = np.squeeze(ac_x_test[img_idx,:,:,:])    
    img_with_dots = helpers.draw_boxes(original_img, pd_points_x,pd_points_y,ac_points_x,ac_points_y)
    
    ##>>>>>>>>>>>>>>> End <<<<<<<<<<<<<<<<<<<<<<<<<<
   return img_with_dots
'''
def draw_boxes(image, pd_points_x,pd_points_y,ac_points_x,ac_points_y):
    
    image_h, image_w,_ = image.shape

    for x,y in zip(pd_points_x,pd_points_y):
        cv2.putText(image,'.',(int(x),int(y)), font, 1,(0,255,0))

    for x,y in zip(ac_points_x,ac_points_y):
        cv2.putText(image,'.',(int(x),int(y)), font, 1,(0,0,255))
 
    return image  


def one_hot_it(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes

    # Arguments
        label: The 2D array segmentation image label
        label_values
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    # st = time.time()
    # w = label.shape[0]
    # h = label.shape[1]
    # num_classes = len(class_dict)
    # x = np.zeros([w,h,num_classes])
    # unique_labels = sortedlist((class_dict.values()))
    # for i in range(0, w):
    #     for j in range(0, h):
    #         index = unique_labels.index(list(label[i][j][:]))
    #         x[i,j,index]=1
    # print("Time 1 = ", time.time() - st)

    # st = time.time()
    # https://stackoverflow.com/questions/46903885/map-rgb-semantic-maps-to-one-hot-encodings-and-vice-versa-in-tensorflow
    # https://stackoverflow.com/questions/14859458/how-to-check-if-all-values-in-the-columns-of-a-numpy-matrix-are-the-same
    semantic_map = []
    for colour in label_values:
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    # print("Time 2 = ", time.time() - st)

    return semantic_map
    
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,1])

    # for i in range(0, w):
    #     for j in range(0, h):
    #         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
    #         x[i, j] = index

    x = np.argmax(image, axis = -1)
    return x


def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values
        
    # Returns
        Colour coded image for segmentation visualization
    """

    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,3])
    # colour_codes = label_values
    # for i in range(0, w):
    #     for j in range(0, h):
    #         x[i, j, :] = colour_codes[int(image[i, j])]
    
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

# class_dict = get_class_dict("CamVid/class_dict.csv")
# gt = cv2.imread("CamVid/test_labels/0001TP_007170_L.png",-1)
# gt = reverse_one_hot(one_hot_it(gt, class_dict))
# gt = colour_code_segmentation(gt, class_dict)

# file_name = "gt_test.png"
# cv2.imwrite(file_name,np.uint8(gt))

def HSV_image_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values
        
    # Returns
        Colour coded image for segmentation visualization
    """

    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,3])
    # colour_codes = label_values
    # for i in range(0, w):
    #     for j in range(0, h):
    #         x[i, j, :] = colour_codes[int(image[i, j])]
    h = np.argmax(image, axis = -1)
    v = np.amax(image, axis = -1) * 255
    s = np.ones(h.shape)*255
    colour_codes = 255 / len(label_values)
    h = h * colour_codes
    h = np.expand_dims(h, axis=2)
    s = np.expand_dims(s, axis=2)
    v = np.expand_dims(v, axis=2)
    print(h.shape)
    x = np.concatenate([h, s, v],2)
    

    return x
