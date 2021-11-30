import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np
from os.path import join as join_path

#from utils import utils as utils
from utils import helpers as helpers
from utils import heatmap_utils as htm_utils
#from utils import svs_utils as svs_utils
#from builders import model_builder
from wsi_utils import hpf_patches_utils_ts
from wsi_utils import svs_utils_final_one as svs_utils
#from wsi_utils import svs_utils_final_one_v2 as svs_utils_v2

from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import jaccard_similarity_score
import matplotlib.pyplot as plt

from models import R2UNet as seg_models

import seaborn as sn
import shutil
import json
from PIL import Image
import scipy.ndimage as ndimage
kernel = np.ones((3,3), np.uint8) 
font = cv2.FONT_HERSHEY_SIMPLEX
from tensorflow import keras

import pdb

valid_images = ['.svs','.jpg','.png']

def create_heatmap(im_map, im_cloud, kernel_size=(3,3),colormap=cv2.COLORMAP_JET,a1=0.5,a2=0.5):
    '''
    img is numpy array
    kernel_size must be odd ie. (5,5)
    ''' 
    # create blur image, kernel must be an odd number
    im_cloud_blur = cv2.GaussianBlur(im_cloud,kernel_size,0)
    im_cloud_clr = cv2.applyColorMap(im_cloud_blur, colormap)
    return (a1*im_map + a2*im_cloud_clr).astype(np.uint8) 
    
parser = argparse.ArgumentParser()
parser.add_argument('--project_name_segmentation', type=str, default="Seatosis_seg_R2UNet", help='Name of your project')
parser.add_argument('--project_steatosis_classification', type=str, default="project_steatosis_ResNet50_BBBL", help='Name of your project')
parser.add_argument('--project_inflamation_classification', type=str, default="project_inflamation_ResNet50", help='Name of your project')

parser.add_argument('--HPF_height', type=int, default=1024, help='Height of cropped input image to network') #1152
parser.add_argument('--HPF_width', type=int, default=1024, help='Width of cropped input image to network')
parser.add_argument('--seg_height', type=int, default=256, help='Height of cropped input image to network')
parser.add_argument('--seg_width', type=int, default=256, help='Width of cropped input image to network')
parser.add_argument('--clas_height', type=int, default=256, help='Height of cropped input image to network')
parser.add_argument('--clas_width', type=int, default=256, help='Width of cropped input image to network')
parser.add_argument('--model_seg', type=str, default="R2UNet", help='The model you are using. See model_builder.py for supported models')
parser.add_argument('--input_svs_image_path', type=str, default="/home/mza/Desktop/MedPace_projects/steatosis_detection_project/steatosis_seg_project/database/wsis/test/", help='Dataset you are using.')

args = parser.parse_args()
print("creating all the necessary diretoreis:")
project_log_path_seg = join_path('experimental_logs/', args.project_name_segmentation+'/')   
training_log_saving_path_seg = join_path(project_log_path_seg,'training/')
testing_log_saving_path_seg_1 = join_path(project_log_path_seg,'testing/')
weight_loading_path_seg = join_path(project_log_path_seg,'weights/')

project_log_path_clas_steatosis = join_path('experimental_logs/', args.project_steatosis_classification+'/')   
training_log_saving_path_clas_steatosis = join_path(project_log_path_clas_steatosis,'training/')
testing_log_saving_path_clas_steatosis = join_path(project_log_path_clas_steatosis,'testing/')
trained_model_steatosis = join_path(project_log_path_clas_steatosis,'trained_model/')
weight_loading_path_clas_steatosis = join_path(project_log_path_clas_steatosis,'weights/')

project_log_path_clas_inflamation = join_path('experimental_logs/', args.project_inflamation_classification+'/')   
training_log_saving_path_clas_inflamation = join_path(project_log_path_clas_inflamation,'training/')
testing_log_saving_path_clas_inflamation = join_path(project_log_path_clas_inflamation,'testing/')
trained_model_inflamation= join_path(project_log_path_clas_inflamation,'trained_model/')

# create pathces from the WSI
HPFs_size = (args.HPF_height,args.HPF_width)  
# Extract specific size of the patches...
#patches_dir = svs_utils.extract_same_size_patches_from_wsi_final(args.input_svs_image_path, testing_log_saving_path_seg_1, HPFs_size)

#pdb.set_trace()
# Extract all of the patches from WSI
patches_dir = '/home/mza/Desktop/MedPace_projects/steatosis_detection_project/steatosis_seg_project/experimental_logs/Seatosis_seg_R2UNet/testing/2MGY7D24/'
slide_name = patches_dir.split('/')[-2]

#pdb.set_trace()

if not os.path.isdir("%s/%s"%(testing_log_saving_path_seg_1,slide_name+'_outputs')):
    os.makedirs("%s/%s"%(testing_log_saving_path_seg_1,slide_name+'_outputs'))        

testing_log_saving_path_seg_2 = join_path(testing_log_saving_path_seg_1,slide_name+'_outputs')

# saving all of the ouput patches...
if not os.path.isdir("%s/%s"%(testing_log_saving_path_seg_2,'output_patches/')):
    os.makedirs("%s/%s"%(testing_log_saving_path_seg_2,'output_patches/'))   
testing_log_saving_path_seg_patcehs = join_path(testing_log_saving_path_seg_2,'output_patches/') 

if not os.path.isdir("%s/%s"%(testing_log_saving_path_seg_2,'output_images/')):
    os.makedirs("%s/%s"%(testing_log_saving_path_seg_2,'output_images/'))   
testing_log_saving_path_seg_images = join_path(testing_log_saving_path_seg_2,'output_images/') 


if not os.path.isdir("%s/%s"%(testing_log_saving_path_seg_2,'FP_images/')):
    os.makedirs("%s/%s"%(testing_log_saving_path_seg_2,'FP_images/'))   
testing_log_saving_path_FP_images = join_path(testing_log_saving_path_seg_2,'FP_images/') 

# Find the json log fle for patches and copy to
#json_file_name = [x for x in sorted(os.listdir(patches_dir)) if x[-5:] == '.json']
#json_file_name_with_dir = join_path(patches_dir,json_file_name[0]) 
#shutil.move(json_file_name_with_dir,testing_log_saving_path_seg_patcehs)

# Model input size for Segmentation model...
seg_net_input = (args.seg_height, args.seg_width,3)
num_classes = 2
print("Model building...")
model_segmentation = seg_models.build_R2UNetED_final(seg_net_input,num_classes)
model_segmentation.summary()
print('-'*30)
print('Loading mean and std for the segmentation and detection models...')
print('-'*30)
# load mean and std for sample normalization for segmentation model
mean_path_grady = join_path(training_log_saving_path_seg,'steatosis_mean.npy')
std_path_grady = join_path(training_log_saving_path_seg,'steatosis_std.npy')
# load weights to model
print('-'*30)
print('Loading weights for all model...')
print('-'*30)
# for Segmentation model
weights = os.listdir(weight_loading_path_seg)[0]
weight_path = join_path(weight_loading_path_seg,weights)
model_segmentation.load_weights(weight_path) 
# Model input size for classification model...
channels = 3
num_classes = 5
clas_net_input = (args.clas_height, args.clas_width,channels)
# load model here..
model_loading_path = join_path(trained_model_steatosis,'model-037-0.939595-0.962956.h5')
model_classification = keras.models.load_model(model_loading_path)
model_classification.summary()

model_loading_path_inflm = join_path(trained_model_inflamation,'model-inflammation_cls_mean_out.h5')
#model_loading_path_inflm = join_path(trained_model_inflamation,'model_IFD_best_model_medpace_public_DB.h5')
#model_loading_path_inflm = join_path(trained_model_inflamation,'model-028-0.889000-0.876248_CycleGAN_dataset.h5')
model_classificationh_inflm = keras.models.load_model(model_loading_path_inflm)
model_classificationh_inflm.summary()

#pdb.set_trace()
# Load the testing data from data directory...
print("Loading image from the directorty ...")              
images_name = [x for x in sorted(os.listdir(patches_dir)) if x[-4:] == '.jpg' or '.png' or '.svs' or '.tif']
print('Total number of images:'+str(len(images_name)))
Total_steatosis_cell_wsi = 0
Total_pixels = 0
total_segmentated_pixels_wsi = 0

#pdb.set_trace()

for i, img_name in enumerate(images_name):
    
    ext = os.path.splitext(img_name)[1]    
    img_name_wo_ext = os.path.splitext(img_name)[0]
    img_path = os.path.join(patches_dir,img_name)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)#.astype("int16").astype('float32')
    gray_img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    patch_h = args.seg_height
    patch_w = args.seg_width
    
    original_img = img   
    # Extract patches (128x128) for segmentation and detection...
    patch_h = args.seg_height
    patch_w = args.seg_width
    img_patches, img_patches_number, num_rows,num_columns =hpf_patches_utils_ts.extract_patches_from_image(img,patch_h,patch_w)
    #gray_img_patches, gray_img_patches_number, gray_num_rows,gray_num_columns =dataset_utils_ts.extract_patches_from_image(gray_img,patch_h,patch_w)
    mask_patches_number  = img_patches_number
    
    ac_x_test = img_patches
    x_test = ac_x_test
    #y_test = mask_patches
    # Normalize the input images with respect to the existing mean and std for segmentation task..
    mean_grady = np.load(mean_path_grady)
    std_grady = np.load(std_path_grady)
    x_test_seg = x_test.astype('float32')
    x_test_seg -= mean_grady
    x_test_seg /= std_grady 
    #y_test_seg /= 255.  # scale masks to [0, 1]
    x_test_seg = x_test_seg.reshape(x_test_seg.shape[0], args.seg_height, args.seg_width,3)
    # apply segmentation model...
    t1 = time.time()
    y_hat_seg = model_segmentation.predict(x_test_seg)
    t2=time.time()                   
    print ("Total time for segmentation:")
    t_f = t2-t1
    print("Total time:",t_f)                   
    number_test_samples = x_test_seg.shape
    y_hat_seg = np.squeeze(y_hat_seg)
    # apply thresholding...
    pred_masks_seg = 255.0*(y_hat_seg[:,:,:] >= 0.8)               
    # Reconstruct original image from the patches
    reconstructed_image = hpf_patches_utils_ts.image_from_patches(ac_x_test, img_patches_number, num_rows,num_columns)
    reconstructed_pred_seg = hpf_patches_utils_ts.image_from_patches(pred_masks_seg, mask_patches_number, num_rows,num_columns)
    #reconstructed_pred_seg_morph_final = 255.0*(reconstructed_pred_seg[:,:] >= 0.5) 
    #mask_gray_img = reconstructed_pred_seg_morph_final
    # with morphological operations.....
    reconstructed_pred_seg_morph = helpers.perform_morphological_operations(reconstructed_pred_seg)            
    reconstructed_pred_seg_morph_final = 255.0*(reconstructed_pred_seg_morph[:,:] >= 0.5) 
    mask_gray_img = reconstructed_pred_seg_morph

    # classification task start from here >>>>>>>>>>>>>>>>>. **********<<<<<<<<<<<<<<<<<<<<<<
    #original_img = img     
      
    img_4_conf = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)    
    img_4_conf = cv2.cvtColor(img_4_conf, cv2.COLOR_RGB2BGR)
                        
    # Extract patches (32x32) for classifiaiton
    patch_h = args.clas_height
    patch_w = args.clas_width            
    #pdb.set_trace()
    img_patches, img_patches_number, num_rows,num_columns, start_x,start_y =hpf_patches_utils_ts.extract_patches_from_image_mask_testing(img_4_conf,mask_gray_img,patch_h,patch_w)
    #gray_img_patches, gray_img_patches_number, gray_num_rows,gray_num_columns =dataset_utils_ts.extract_patches_from_image(gray_img,patch_h,patch_w)
    mask_patches_number  = img_patches_number            
    ac_x_test = img_patches
    x_test = ac_x_test            
    x_test_class = img_patches.reshape(img_patches.shape[0], args.clas_height, args.clas_width,3)
    # Preprocessing test image...
    #y_test_seg /= 255.  # scale masks to [0, 1]    
    x_test_class_steatotis = hpf_patches_utils_ts.preprocess_input_steatosis(x_test_class)
    t1 = time.time()
    y_hat_class = model_classification.predict(x_test_class_steatotis)
    t2=time.time()
    
    x_test_class_inflm = hpf_patches_utils_ts.preprocess_input_inflm(x_test_class)
    y_hat_class_inflm = model_classificationh_inflm.predict(x_test_class_inflm)
      
    print ("Total time for segmentation:")
    t_f = t2-t1
    print("Total time:",t_f)                                
    number_test_samples = x_test_class.shape                                  
    num_samples = len(y_hat_class)
            
    final_start_x = []
    final_start_y = []
    max_conf_values = []
    class_indexes = []
    max_conf_values_inflm = []
    class_indexes_inflm = []
    ## Setup a thrshold to classify normal or malignants...    
    #pdb.set_trace()
    thresh_value = 0.92                
    if num_samples>0:                
        for k in range(num_samples):
            ## class information for steatosis
            conf_arr = y_hat_class[k,:]
            conf_arr = np.array(conf_arr)
            conf_value = np.amax(conf_arr)                    
            #if conf_value > thresh_value:
            conf_index = np.where(conf_arr == np.amax(conf_arr))
            max_conf_values.append(conf_value)
            class_indexes.append(conf_index[0][0])
            
            ## class information for inflamation
            conf_arr_inflm = y_hat_class_inflm[k,:]
            conf_arr_inflm = np.array(conf_arr_inflm)
            conf_value_inflm = np.amax(conf_arr_inflm)                    
            #if conf_value > thresh_value:
            conf_index_inflm = np.where(conf_arr_inflm == np.amax(conf_arr_inflm))
            max_conf_values_inflm.append(conf_value_inflm)
            class_indexes_inflm.append(conf_index_inflm[0][0])
        
    logs = zip(start_x,start_y,max_conf_values,class_indexes)      
    logs_inflm = zip(start_x,start_y,max_conf_values,class_indexes)      
    ## save logs file for classifier...
    #pdb.set_trace()
    # save logs for entire wsi.....
    log_classifier_hpf = img_name_wo_ext+'_clsf_logs_steatosis.txt'
    patch_classifier_logs = join_path(testing_log_saving_path_seg_patcehs,log_classifier_hpf)        
    class_hpf_logs = open(patch_classifier_logs, 'w')
    #overall_streatosis_per = (total_segmentated_pixels_wsi /Total_pixels)*100
    # 
    class_hpf_logs.write("HPF_id: "+str(img_name_wo_ext)
                        + "\n conf_values: " +str(max_conf_values)
                        + "\n class_index :"+str(class_indexes)
                        #+ "\n Steatosis in percentage:"+str(overall_streatosis_per)
                        )
    log_classifier_inflm = img_name_wo_ext+'_clsf_logs_inflamation.txt'
    patch_classifier_logs_inflm = join_path(testing_log_saving_path_seg_patcehs,log_classifier_inflm)        
    clfr_logs_inflm = open(patch_classifier_logs_inflm, 'w')
    #overall_streatosis_per = (total_segmentated_pixels_wsi /Total_pixels)*100
    # 
    clfr_logs_inflm.write("HPF_id: "+str(img_name_wo_ext)
                        + "\n conf_values: " +str(max_conf_values_inflm)
                        + "\n class_index :"+str(class_indexes_inflm)
                        #+ "\n Steatosis in percentage:"+str(overall_streatosis_per)
                        )
                              
    #generated_heatmap_image = dataset_utils.generate_heatmap_image_from_conf_values(logs,args.crop_height, args.crop_width,args.HPF_height,args.HPF_width)        
    #generated_heatmap_org_image = dataset_utils.generate_heatmap_image_from_conf_values_on_org(img_4_conf, logs,args.crop_height, args.crop_width,args.HPF_height,args.HPF_width)
    #final_img_name = img_name_wo_ext   
    #pdb.set_trace()  
    #generated_heatmap_org_image_seg_clas = dataset_utils_ts.generate_heatmap_image_from_conf_values_on_org_saving_FP(img_4_conf, logs,args.clas_height, args.clas_width,args.HPF_height,args.HPF_width,final_img_name,testing_log_saving_path_FP_images)        
    #generated_mask_seg_clas,num_pxls_bronchial_cells_HPFs,num_pxls_lyphoctyte_cells_HPFs  =dataset_utils_ts.generate_final_mask_from_seg_class(reconstructed_pred_seg_morph_final, logs, args.clas_height, args.clas_width,args.HPF_height,args.HPF_width)   
    #pdb.set_trace()
    #generated_mask_seg_clas = hpf_patches_utils_ts.generate_final_mask_from_steatosis_seg_class(reconstructed_pred_seg_morph_final, logs, args.clas_height, args.clas_width,args.HPF_height,args.HPF_width)
    #generated_mask_seg_clas = svs_utils_v2.generate_steatosis_final_mask(reconstructed_pred_seg_morph_final, start_x,start_y,max_conf_values,class_indexes, args.clas_height, args.clas_width,args.HPF_height,args.HPF_width)
    
    ## >>>>>>>>>>>>>>>>>>>>>>>>>>>    final mask for steatosis detection from seg and classification <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    final_maska_steatosis = np.zeros((args.HPF_height,args.HPF_width),dtype=int)
    for k in range(len(start_x)):        
        row = int(start_x[k]*patch_h)
        column = int(start_y[k]*patch_w)
        index_value = int(class_indexes[k])
        if index_value <= 3:  
            indv_mask = reconstructed_pred_seg_morph_final [row:row+patch_h, column: column+patch_w]
            final_maska_steatosis [row:row+patch_h, column: column+patch_w ] = indv_mask            
        else:
            indv_mask = reconstructed_pred_seg_morph_final [row:row+patch_h, column: column+patch_w]
            indv_mask = (indv_mask/255)*0              
            final_maska_steatosis [row:row+patch_h, column: column+patch_w ] = indv_mask 
    
    #pdb.set_trace()        
    final_maska_steatosis = 255.0*(final_maska_steatosis[:,:] >= 0.5) 
    
    ## >>>>>>>>>>>>>>>>>>>>>>>>>>>    final mask for inflamation detection from seg and classification <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    final_maska_inflamation = np.zeros((args.HPF_height,args.HPF_width),dtype=int)
    for k in range(len(start_x)):        
        row = int(start_x[k]*patch_h)
        column = int(start_y[k]*patch_w)
        index_value = int(class_indexes_inflm[k])
        idv_conf_val_iflm = max_conf_values_inflm[k]
        if index_value == 1 and idv_conf_val_iflm > 0.5:  
            print('Inflamation detected in :'+str(img_name_wo_ext)+' and'+' conf. values:'+str(idv_conf_val_iflm))
            indv_mask = reconstructed_pred_seg_morph_final [row:row+patch_h, column: column+patch_w]
            final_maska_inflamation [row:row+patch_h, column: column+patch_w ] = indv_mask            
        else:
            indv_mask = reconstructed_pred_seg_morph_final [row:row+patch_h, column: column+patch_w]
            indv_mask = (indv_mask/255)*0              
            final_maska_inflamation [row:row+patch_h, column: column+patch_w ] = indv_mask 
    
    #pdb.set_trace()        
    final_maska_steatosis = 255.0*(final_maska_steatosis[:,:] >= 0.5)  
    final_maska_inflamation = 255.0*(final_maska_inflamation[:,:] >= 0.5)  

    # circular blobs detection 
    #pdb.set_trace()
    #generated_mask_seg_clas = 255-(255*final_refined_mask)  
    generated_mask_seg_clas = final_maska_steatosis
    #generated_heatmap_image= np.array(final_mask)
    uniques_values = np.unique(generated_mask_seg_clas)
    print(uniques_values)         

    # calculate the steatosis_pixel in HPF and accumulate for WSI
    num_pxls_steatosis_cells_HPFs = np.sum((generated_mask_seg_clas)==255)
    print('Num_pxls_steatosis_cells_HPFs  :'+str(num_pxls_steatosis_cells_HPFs))
    Total_steatosis_cell_wsi = Total_steatosis_cell_wsi+num_pxls_steatosis_cells_HPFs

    # calculate the total number of for HPF and accumulate for generating WSI   
    pixel_HPF_total = args.HPF_height*args.HPF_width
    Total_pixels = Total_pixels + pixel_HPF_total
    
    # Calculate the overall percentage of steatosis in HPF.....
    steatosis_per_hpf = (num_pxls_steatosis_cells_HPFs /pixel_HPF_total)*100
    #total_segmented_pixels_HPF = Total_steatosis_cell_wsi
    #total_segmentated_pixels_wsi = total_segmentated_pixels_wsi+total_segmented_pixels_HPF
    
    log_file_name = img_name_wo_ext+'_pixels_steatosis_seg_HPFs_logs.txt'
    patch_output_logs = join_path(testing_log_saving_path_seg_patcehs,log_file_name)        
    patch_logs = open(patch_output_logs, 'w')

    patch_logs.write("Total_pixels in HPF: "+str(pixel_HPF_total)
                        + "\n Total steatosis cells in HPF : " +str(num_pxls_steatosis_cells_HPFs)
                        + "\n Steatosis pixels in percentage:"+str(steatosis_per_hpf)

                        )                
                                
    print('Processing for:'+str(img_name_wo_ext))           
    # saving output images...
    ac_img_name =img_name_wo_ext+'_actual_img.jpg'
    y_pred_name_seg = img_name_wo_ext+'_image_seg'+'.jpg'  
    y_pred_name_seg_morph_st = img_name_wo_ext+'_image_seg_class_st'+'.jpg' 
    y_pred_name_seg_morph_infm = img_name_wo_ext+'_image_seg_class_infm'+'.jpg'    

    #y_pred_name_seg_morph_refine_mask = img_name_wo_ext+'_image_seg_class_refine_mask'+'.jpg'    
    
    final_des_img = os.path.join(testing_log_saving_path_seg_patcehs,ac_img_name)
    final_des_pred_seg = os.path.join(testing_log_saving_path_seg_patcehs,y_pred_name_seg)
    final_des_pred_seg_morph_st = os.path.join(testing_log_saving_path_seg_patcehs,y_pred_name_seg_morph_st)
    final_des_pred_seg_morph_infm = os.path.join(testing_log_saving_path_seg_patcehs,y_pred_name_seg_morph_infm)

    cv2.imwrite(final_des_img,original_img)
    cv2.imwrite(final_des_pred_seg,reconstructed_pred_seg)
    cv2.imwrite(final_des_pred_seg_morph_st,generated_mask_seg_clas)
    cv2.imwrite(final_des_pred_seg_morph_infm,final_maska_inflamation)

    #cv2.imwrite(final_des_pred_seg_morph,final_refined_mask)
    #if reconstructed_pred_seg_morph_final.max() > 0: 
    #    cv2.imwrite(final_des_pred_seg_morph_refine_mask,refined_heatmap_image)
    

# save logs for entire wsi.....
#log_file_name_for_wsi = slide_name+'_pixels_steatosis_cells_logs_wsi.txt'
#patch_output_logs = join_path(testing_log_saving_path_seg_images,log_file_name_for_wsi)        
#patch_logs = open(patch_output_logs, 'w')
#overall_streatosis_per = (total_segmentated_pixels_wsi /Total_pixels)*100
# 
#patch_logs.write("Total_pixels: "+str(Total_pixels)
#                    + "\n Total steatosis cells in WSI: " +str(total_segmentated_pixels_wsi)
#                    + "\n Total segmented pixels in WSI:"+str(total_segmentated_pixels_wsi)
#                    + "\n Steatosis in percentage:"+str(overall_streatosis_per)
#                    )

#pdb.set_trace()

## cleate the WSI-level outputs ........
#svs_utils.patches_to_actual_image_and_ROI_refined_mask_medpace(testing_log_saving_path_seg_patcehs,testing_log_saving_path_seg_images)
#svs_utils.patches_to_binary_image_from_seg_class_masks_medpace_final(testing_log_saving_path_seg_patcehs,testing_log_saving_path_seg_images)
### Create heatmpas.... 
## create heatmaps outputs from roi from seg+class masks
##svs_utils.create_heatmaps_from_roi_images_from_class_plus_seg_mask(testing_log_saving_path_seg_images,testing_log_saving_path_seg_images)
## create heatmaps outputs from roi mask and segmentation (seg+class) masks
#svs_utils.create_heatmaps_from_roi_images_from_combined_seg_mask(testing_log_saving_path_seg_images,testing_log_saving_path_seg_images)
#  
#   