#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 10:44:47 2021

@author: mza
"""

import numpy as np
from wsi_utils import svs_utils_final_one as svs_utils
#from wsi_utils import svs_utils_final_one_v2 as svs_utils_v2

import pdb
#pdb.set_trace()

HPF_height = 1024
HPF_width = 1024

#input_svs_image_path = '/home/mza/Desktop/MedPace_projects/steatosis_detection_project/steatosis_seg_project/database/wsis/test/'

testing_log_saving_path_seg_patcehs = '/home/mza/Desktop/MedPace_projects/steatosis_detection_project/steatosis_seg_project/experimental_logs/Seatosis_seg_R2UNet/testing/2MGY7D24_outputs/output_patches/'
testing_log_saving_path_seg_images = '/home/mza/Desktop/MedPace_projects/steatosis_detection_project/steatosis_seg_project/experimental_logs/Seatosis_seg_R2UNet/testing/2MGY7D24_outputs/output_images/'


#pdb.set_trace()
svs_utils.patches_to_actual_image_and_ROI_refined_mask_medpace(testing_log_saving_path_seg_patcehs,testing_log_saving_path_seg_images)
svs_utils.patches_to_binary_image_from_seg_class_steatosis_masks_medpace_final(testing_log_saving_path_seg_patcehs,testing_log_saving_path_seg_images)
svs_utils.patches_to_binary_image_from_seg_class_inflamation_masks_medpace_final(testing_log_saving_path_seg_patcehs,testing_log_saving_path_seg_images)
## Create heatmpas.... 
# create heatmaps outputs from roi from seg+class masks
#svs_utils.create_heatmaps_from_roi_images_from_class_plus_seg_mask(testing_log_saving_path_seg_images,testing_log_saving_path_seg_images)
svs_utils.create_heatmaps_from_roi_images_from_class_plus_seg_mask_st(testing_log_saving_path_seg_images,testing_log_saving_path_seg_images)
svs_utils.create_heatmaps_from_roi_images_from_class_plus_seg_mask_infm(testing_log_saving_path_seg_images,testing_log_saving_path_seg_images)

# create heatmaps outputs from roi mask and segmentation (seg+class) masks
#svs_utils.create_heatmaps_from_roi_images_from_combined_seg_mask(testing_log_saving_path_seg_images,testing_log_saving_path_seg_images)

  
  