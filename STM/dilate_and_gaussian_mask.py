global_iter = 20  # Change these numbers to adjust gaussian blur & dilation 
global_sigma = 30


import glob
import os.path as osp
import sys
import os
import json
import ipdb
from PIL import Image
from PIL.Image import fromarray as ifa
import numpy as np
from numpy import array as npa
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import gaussian_filter
import skimage

def dilate_and_gaussian_mask(mask_img):
    '''
    mask_img:  PIL IMG, shape (height, width), unique = [0,255]
    return: mask, PIL
    '''
    array_dillation = npa(mask_img)/255
    array_dillation = array_dillation.astype(np.bool)
    array_dillation = binary_dilation(array_dillation.astype(np.bool), iterations=global_iter)
    array_dillation = array_dillation.astype(np.uint8)*255

    array_dillation = gaussian_filter(array_dillation, sigma=global_sigma)
    #array_dillation = array_dillation
    mask_dilated = ifa(array_dillation)
    #mask_dilated = Image.fromarray( binary_dilation(np.array(mask_img)/255, iterations=1))
    return mask_dilated

def process_input_image(mask_t_1, jpeg_t, flow_t_1=None):
    '''
    mask_t_1: PIL Image, mask at time step t-1
    jpeg_t: PIL Image, image at time step t
    flow_t_1: flo
    return: Jpeg PIL Image
    '''
    mask_t_1_dilated = npa(dilate_and_gaussian_mask(mask_t_1))
    jpeg_t = npa(jpeg_t)
    #jpeg_t = skimage.util.invert(jpeg_t) #INVERT
    jpeg_t = jpeg_t *np.expand_dims( (mask_t_1_dilated/255),2)
    jpeg_t = jpeg_t.astype(np.uint8)
    return ifa(jpeg_t)

if __name__=="__main__":
    video_name = "bike-trial_01"
    frame_id = "00002" # image at frame t
    mask_id  = "00001" # mask at frame t - 1
    t_1_mask = osp.join("stm-mask-instances/", video_name, frame_id + ".png")
    image_path = osp.join("../DAVIS-test-challenge/JPEGImages/480p/", video_name.split('_')[0], frame_id + ".jpg")
    
    # Load images & masks
    image_content =  Image.open(image_path)
    mask_t_1_content = Image.open(t_1_mask)
    
    # Dilate and gaussian mask for frame t
    out_image_blured_t = process_input_image(mask_t_1 = mask_t_1_content, jpeg_t = image_content)
    
    # Save output
    output_dir = "./blur_out"
    output_mask_dir = osp.join(output_dir, video_name)
    if not osp.exists(output_dir): os.makedirs(output_dir)
    if not osp.exists(output_mask_dir): os.makedirs(output_mask_dir)
    out_image_blured_t.save(osp.join(output_mask_dir, frame_id + ".jpg"))