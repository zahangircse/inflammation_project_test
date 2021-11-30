#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 17:13:51 2021

@author: mza
"""
import cv2
import numpy as np


def create_heatmap(im_map, im_cloud, kernel_size=(3,3),colormap=cv2.COLORMAP_JET,a1=0.5,a2=0.5):
    '''
    img is numpy array
    kernel_size must be odd ie. (5,5)
    ''' 
    # create blur image, kernel must be an odd number
    im_cloud_blur = cv2.GaussianBlur(im_cloud,kernel_size,0)
    im_cloud_clr = cv2.applyColorMap(im_cloud_blur, colormap)
    return (a1*im_map + a2*im_cloud_clr).astype(np.uint8) 