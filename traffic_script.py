#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This script was designed to be an example of practical use for the model developed in the project:
"Highly Congested Scenes Analysis (i.e.: Traffic)"
https://github.com/jorgealcubilla/TFM_DataScience

The purpose of this script is to make estimations of traffic density, using a machine learning model
and based on a given traffic image.

The code of this script has been structured as follows:
    # Python libraries required
    # Definition of functions to be used by this script
    # Setting function paramenters (paramenters used in the model trainning process) 
    # Paths setting
    # Script processing code

User manual:
***This script has been developed to be run using Python3***

This script collects images of traffic from a webpage and returns:
- ".txt" file with the traffic density estimation of the last image processed.
- History dataset, including: collected images and a "csv" file with their corresponding traffic density estimations (values)

Previously, the following steps must be followed:

1) Setup the process for image file collection:  link to webpage image 
WARNING: 
    - The path to the image to be collected MUST BE ALWAYS THE SAME.
    - This script is design to collect an image from a website using "urllib". Other sources can be set by tweaking
      the code in section "Script processing code".

2) Create a mask to retrict the image to a region of interest (relevant area of the image) with and save it.
The image that will be analyze by the machine learning model must be restricted to the area where traffic takes place.
WARNING: This script was designed to use a MATLAB mask. Any binary mask will be suitable, it only requires to tweak
the code to read the corresponding file. This code is located in "Script processing code" section.

3) Save the parameters for the machine learning model. They are provided by the project mentioned above.
The name of the file is: "model_w.h5"

4) Set paths to already mentioned files (image, mask and parameters) and set:
- Last image estimation: path to the folder where file with the estimation of traffic density will be saved and give name to the file
- History dataset: path to the folder where this dataset will be saved and give name to the ".csv" file where the
  history of estimations will be stored.
WARNING: Go to "Paths setting" section to complete this step.

5) Define time zone (if different to the local one): Go to the last block of code in "Script processing code" section
WARNING: the time zone set is 'US/Pacific'. Please, set the one that better suits your needs.

6) Set range of prediction values for each level of traffic density ("High", Medium" and "Low"): 
   Go to the last block of code in "Script processing code" section

7) Run this script with Python3

'''
#########################################################################################################################
# Python libraries required
#########################################################################################################################

import numpy as np
import time

import skimage.io
import scipy.io as sio
from scipy.ndimage.filters import gaussian_filter 
from skimage.transform import resize
import urllib
import datetime
import os

from keras.preprocessing import image as kimage

#To create Sequential model:
from keras import models

#To add layers to the model:
from keras import layers

from git import Repo,remote
from pytz import timezone

#########################################################################################################################
# Definition of functions to be used by this script
#########################################################################################################################

# Machine learning model for density map predictions ("3-head Hydra_CNN"):

def hydra_model(): 
### CCNN_s0:
    head0_input = layers.Input(shape=(3,72,72))
    head0 = layers.ZeroPadding2D(padding=(3, 3), data_format="channels_first")(head0_input)
    head0 = layers.Conv2D(32, kernel_size=7, activation='relu', name="head0_conv1", data_format="channels_first")(head0)
    head0 = layers.MaxPooling2D(pool_size=2, strides=2, data_format="channels_first")(head0)

    head0 = layers.ZeroPadding2D(padding=(3, 3), data_format="channels_first")(head0)
    head0 = layers.Conv2D(32, kernel_size=7, activation='relu', name="head0_conv2", data_format="channels_first")(head0)
    head0 = layers.MaxPooling2D(pool_size=2, strides=2, data_format="channels_first")(head0)

    head0 = layers.ZeroPadding2D(padding=(2, 2), data_format="channels_first")(head0)
    head0 = layers.Conv2D(64, kernel_size=5, activation='relu', name="head0_conv3", data_format="channels_first")(head0)

    head0 = layers.Conv2D(1000, kernel_size=1, activation='relu', name="head0_conv4", data_format="channels_first")(head0)

    head0 = layers.Conv2D(400, kernel_size=1, activation='relu', name="head0_conv5", data_format="channels_first")(head0)

### CCNN_s1:
    head1_input = layers.Input(shape=(3,72,72))
    head1 = layers.ZeroPadding2D(padding=(3, 3), data_format="channels_first")(head1_input)
    head1 = layers.Conv2D(32, kernel_size=7, activation='relu', name="head1_conv1", data_format="channels_first")(head1)
    head1 = layers.MaxPooling2D(pool_size=2, strides=2, data_format="channels_first")(head1)

    head1 = layers.ZeroPadding2D(padding=(3, 3), data_format="channels_first")(head1)
    head1 = layers.Conv2D(32, kernel_size=7, activation='relu', name="head1_conv2", data_format="channels_first")(head1)
    head1 = layers.MaxPooling2D(pool_size=2, strides=2, data_format="channels_first")(head1)

    head1 = layers.ZeroPadding2D(padding=(2, 2), data_format="channels_first")(head1)
    head1 = layers.Conv2D(64, kernel_size=5, activation='relu', name="head1_conv3", data_format="channels_first")(head1)

    head1 = layers.Conv2D(1000, kernel_size=1, activation='relu', name="head1_conv4", data_format="channels_first")(head1)

    head1 = layers.Conv2D(400, kernel_size=1, activation='relu', name="head1_conv5", data_format="channels_first")(head1)

### CCNN_s2:
    head2_input = layers.Input(shape=(3,72,72))
    head2 = layers.ZeroPadding2D(padding=(3, 3), data_format="channels_first")(head2_input)
    head2 = layers.Conv2D(32, kernel_size=7, activation='relu', name="head2_conv1", data_format="channels_first")(head2)
    head2 = layers.MaxPooling2D(pool_size=2, strides=2, data_format="channels_first")(head2)

    head2 = layers.ZeroPadding2D(padding=(3, 3), data_format="channels_first")(head2)
    head2 = layers.Conv2D(32, kernel_size=7, activation='relu', name="head2_conv2", data_format="channels_first")(head2)
    head2 = layers.MaxPooling2D(pool_size=2, strides=2, data_format="channels_first")(head2)

    head2 = layers.ZeroPadding2D(padding=(2, 2), data_format="channels_first")(head2)
    head2 = layers.Conv2D(64, kernel_size=5, activation='relu', name="head2_conv3", data_format="channels_first")(head2)

    head2 = layers.Conv2D(1000, kernel_size=1, activation='relu', name="head2_conv4", data_format="channels_first")(head2)

    head2 = layers.Conv2D(400, kernel_size=1, activation='relu', name="head2_conv5", data_format="channels_first")(head2)


## CCNNs Concatenation & Fully addition of connected layers:
    body = layers.concatenate([head0, head1, head2], axis=1)
    body = layers.Flatten()(body)

    body = layers.Dense(512, activation='relu', name="body_fc6")(body)
    body = layers.Dense(512, activation='relu', name="body_fc7")(body)
    body = layers.Dense(324, activation='relu', name="body_fc8")(body)


    model = models.Model(inputs=[head0_input, head1_input, head2_input], outputs=body)

    
    return model   
    
    
    
def genDensity(dot_im, sigmadots):
    '''
    @brief: This function gets a dotted image and returns its density map.
    @param: dots: annotated position of objects consisting of image of dots.
    @param: sigmadots: density radius.
    @return: density map for the input dots image.   
    '''
    
    # Takes only red channel (where the dots are annotated)
    dot = dot_im[:, :, 0]
    dot = gaussian_filter(dot, sigmadots)
        
    return dot.astype(np.float32)
    

def cartesian(arrays, out=None):
    """
    This function will provide a list of coordinates (positions) required by 'get_dense_pos' function (see below)
    to select those that will be used to build the patches (115x115).
    
    """    
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
            
    return out


def get_dense_pos(heith, width, pw, stride = 1):
    '''
    The patches will be created by selecting points of the image as the center of the patch.
    Thus, the points of the image that generate patches whose coordinates go beyond the image´s perimeter 
    will not be selected.
    
    This function provides a dense list of points that will be used to build the patches (115x115).
    
    Those points will be selected from the area of the image whose points are at a distance from image´s perimeter 
    equal or higher than half of the patch height (or width), with a stride set by the 'stride' parameter 
    (stride=10 in Hydra model).
    
    @param heith: image height.
    @param width: image width.
    @param pw: patch with.
    @param stride: stride.
    
    '''    
    # Computes half a patch (height or width, it´s the same)
    dx=dy=int(pw/2)
    # CCoordinates of area of the image whose points are at a distance from its perimeter equal or higher
    #than half of a patch: 
    pos = cartesian( (range(dx, heith - dx, stride), range(dy, width -dy, stride) ) )
    bot_line = cartesian( (heith - dx -1, range(dy, width -dy, stride) ) )
    right_line = cartesian( (range(dx, heith - dx, stride), width -dy - 1) )
    
    return np.vstack( (pos, bot_line, right_line) )
    
    

def extractEscales(lim, n_scales, input_pw):
    '''
    Builds a 'pyramid' of different scale levels for each extracted patch, and
    resizes the crops (s_0, s_1 and s_2 in our case) to the input size to feed the Hydra_CNN.

    @param lim: list of patches related to the original image
    @param n_scales: number of different scale levels (3 in our case)
    @param input_pw: input size (72x72 pixels)
    
    '''
    out_list = [] #List of 'pyramids' corresponding to all the patches extracted from the original image
    for im in lim:
        ph, pw = im.shape[0:2] # it will get the patch width and height (115x115)
        scaled_im_list = [] # list of crops ('pyramid') related to a specific patch
        
        #Crops generator and resizing:
        for s in range(n_scales):
            ch = int(s * ph / (2*n_scales))
            cw = int(s * pw / (2*n_scales))
            
            crop_im = im[ch:ph-ch, cw:pw - cw]
            
            #s=0, s_0 = original patch = (115x115)
            #s=1, s_1 = 66% of original patch = (77x77)
            #s=2, s_2 = 33% of original patch = (39x39)
            
            #Resizes the crops (s_0, s_1 and s_2 in our case) to the input size to feed the Hydra_CNN (72x72)            
            scaled_im_list.append(resize(crop_im, (input_pw, input_pw)))
        
        out_list.append(scaled_im_list)
        
    return out_list
    
    
def resizeDensityPatch(patch, opt_size):
    '''
    Takes a density map and resizes it to the opt_size.
    
    @param patch: input density map.
    @param opt_size: output size (the original size of the patch, 115x115, in our case).
        
    The rescaling process will generate a density map whose associated count (pixel values addition) will not
    necessarily match the count of the input density map.
    Therefore, the input density map will be normalized before rescaling, reversed after rescaling
    and the final count adjusted.
    
    '''
    # Input normalization to values between 0 and 1:
    patch_sum = patch.sum()   
    p_max = patch.max()
    p_min = patch.min()    
    if patch_sum !=0: # Avoid 0 division
        patch = (patch - p_min)/(p_max - p_min)
    
    # Resizing to the original size of the patch:
    patch = resize(patch, opt_size)
    
    # Normalization reversal:
    patch = patch*(p_max - p_min) + p_min
    
    # Count adjustment:
    res_sum = patch.sum()
    if res_sum != 0:
        return patch * (patch_sum/res_sum)

    return patch
    

##Function to process images with a "3-head Hydra_model" and get their predicted density map: 
def process(model, im, n_scales, base_pw, input_pw):
    
    '''
    Parameters (values based on the paper´s methodology):
    model: model to be used for density map prediction purposes (3-head Hydra_CNN)
    im: image from which a density map will be estimated
    n_scales: number of different scale levels to feed the Hydra_CNN as inputs (for the 3-head Hydra_CNN, 
                there will be 3: s:0, s_1, s_2)
    base_pw: original size of patches to be extracted from the image (115x115 pixels)
    inpu_pw: inputs size (72x72 pixels)
    
    '''
#1) Data preprocessing: original image decomposition into 115x115 overlapped patches
    #Steps:
    
    #1.1) From the original image, extract all the consecutive 115x115 patches with a stride of 10 pixels: 
            
    # Obtaining a dense list of points (coordinates) from the image that will be used to build the patches (115x115)        
    [heith, width] = im.shape[0:2]
    pos = get_dense_pos(heith, width, base_pw, stride=10)

    # Initialize density matrix and votes counting
    dens_map = np.zeros( (heith, width), dtype = np.float32 )   # Init density to 0
    count_map = np.zeros( (heith, width), dtype = np.int32 )    # Number of votes to divide
        
    # Iterate for all patches
    for ix, p in enumerate(pos):# Iterate over all patches               
        dx=dy=int(base_pw/2)    # Compute displacement from centers
        x,y=p
        sx=slice(x-dx, x+dx+1, None)
        sy=slice(y-dy, y+dy+1, None)
        crop_im=im[sx,sy,...]
        h, w = crop_im.shape[0:2]
        if h!=w or (h<=0):
            continue
    #1.2) Build a 'pyramid' of 3 different scale levels ('s') for each extracted patch, and,
    #1.3) Resize s_0, s_1 and s_2 to 72x72 pixels to feed the Hydra_CNN.            
            
        im_scales = extractEscales([crop_im], n_scales, input_pw)
                        
        head0_input = np.expand_dims(im_scales[0][0].copy().transpose(2,0,1), axis=0)
        head1_input = np.expand_dims(im_scales[0][1].copy().transpose(2,0,1), axis=0)
        head2_input = np.expand_dims(im_scales[0][2].copy().transpose(2,0,1), axis=0)
            
#2) 3-head Hydra_CNN processing: Obtaining densitiy map prediction for each 115x115 patch 
            
        pred = model.predict([head0_input,  head1_input,  head2_input])
                       
#3) Density map assembly: Assembly of all the predicted maps to get the density map for the whole original image

    #3.1) Reshape&resize Hydra_CNN´s outputs to get the density map for each patch:
        #Rashape to a 2-dimention array
        p_side = int(np.sqrt( len( pred.flatten() ) )) 
        pred = pred.reshape(  (p_side, p_side) )
            
        # Resize it back to the original patch size (115x115 pixels)
        pred = resizeDensityPatch(pred, crop_im.shape[0:2])          
        pred[pred<0] = 0

    #3.2) Assembly of all patches density map estimations:
        # Predicted density map for each patch is added to the total density map:
        dens_map[sx,sy] += pred
        #Since overlapping occurs, a matrix is Summing up the times (votes) each coordinate has been predicted, 
        #for afterwards average calculation purposes:
        count_map[sx,sy] += 1

    # Remove Zeros
    count_map[ count_map == 0 ] = 1

    # Average density map
    dens_map = dens_map / count_map   
    
     # Final average density map for the whole original image:    
    return dens_map

   
#################################################################################################################
# Setting function paramenters (paramenters used in the model trainning process) 
#################################################################################################################
base_pw = 115
n_scales = 3
sigmadots = 15 # For image density map purposes
input_pw = 72

#################################################################################################################
# Paths setting
#################################################################################################################

##### Machine Learning Model #####
# Root to model´s data (parameters and mask):
root = os.path.abspath("###path to folders where parameters and mask will be saved")
# input image:
img_link = '###path to the image###'
# Model parameters:
weights_path = os.path.join(root, "model_w.h5") # They can be found in folder "model_parameters" 
                                                                    # of the project this script is based on
# Mask to restrict density maps to the ROI (region of interest):
mask_im_path =os.path.join(root, "###file name###")

##### Last image #####
# Root to folder where the last estimation will be saved: 
repo_root = os.path.abspath('###path to the folder### ')  
# Give name to the ".txt" file where the estimation will be saved:
img_density_path = os.path.join(repo_root, "###name of the .txt file###") #The file will be created automatically

##### History Dataset #####
# Path to dataset´s folder:
dataset_path = os.path.abspath("###path to folder where the dataset will be saved###")
# Give name to the "csv" file where the history of estimations will be saved:
csv_path = os.path.join(dataset_path, "###name of the .csv file###") #The file will be created automatically

#################################################################################################################
# Script processing code:
#################################################################################################################

# Remove last estimation before saving a new one
os.remove(img_density_path) 

#Obtain date and time of the last process execution
img_date = datetime.datetime.now()
    
# History Dataset: Name image´s file based on its collection date and sets its path
img_name = 'img_'+str(img_date)[:10]+'_'+str(img_date)[11:13]+str(img_date)[14:16]+'.jpg'
img_path = os.path.join(dataset_path, img_name) # Path to image in the dataset
    
# History Dataset: Collect input image and save it in the dataset´s folder    
urllib.request.urlretrieve(img_link, img_path) 

# Read input image        
im = skimage.img_as_float(skimage.io.imread(img_path)).astype(np.float32)
        
# Read mask that will be applied to density maps to get their ROI (Region of Interest)
mask = sio.loadmat(mask_im_path, chars_as_strings=1, matlab_compatible=1)
mask = mask.get('BW')
    
# Get avarage image density map
hydra = hydra_model() # Sets the CNN model
hydra.load_weights(weights_path, by_name=True) # Load trainned parameters into the model
resImg = process(hydra, im, n_scales, base_pw, input_pw) # Run the model over the input image and get its estimated density map
resImg = resImg * mask # Apply mask to get ROI

# Calculates the predicted vehicles' number 
npred=resImg.sum()

# History Dataset: Image reference
img_number = str(img_path[-19:-4])
    
# History Dataset: Saving prediction data on a 'csv' file
with open(csv_path,'a') as csv_file:
    row = img_number + "," + str(npred) + "," + str(img_date)+"\n"  # History Dataset information: reference, prediction, date_time(local)
    csv_file.write(row)  
csv_file.close()

# Last image: dating and classification of traffic density level

img_date_can = img_date.astimezone(timezone('US/Pacific')) #Converting time zone from local 
                                                           #to image source´s (Vancouver, Canada)
    
with open(img_density_path,'w') as dens: # Translate prediced amount of vehicles into a traffic density level ('low', 'medium', high)
    if int(npred) <= 5:
        dens.write("Traffic Density Estimation: Low "+"(value: "+str(int(npred))+"),\n"+"Last update as of: "+str(img_date_can)[:19]+",\n(Pacific Std. Time)")
    elif 5 < int(npred) <= 10:
        dens.write("Traffic Density Estimation: Medium "+"(value: "+str(int(npred))+"),\n"+"Last update as of: "+str(img_date_can)[:19]+",\n(Pacific Std. Time)")
    elif int(npred) > 10:
        dens.write("Traffic Density Estimation: High "+"(value: "+str(int(npred))+"),\n"+"Last update as of: "+str(img_date_can)[:19]+",\n(Pacific Std. Time)")

dens.close()
    

