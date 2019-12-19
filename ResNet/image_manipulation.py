# import PyTorch, Torchvision packages
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torchvision import transforms

# import Maplotlib
%matplotlib inline
import matplotlib.image as mpimg

# import Numpy
import numpy as np

# import Python standard library
import time
import os,sys
import random

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

# Function that implements random cropping

def uniform(a,b):
    return(a+(b-a)*random.random())

def img_rnd_crop(im, w, h, i = -1, j = -1):
    is_2d = len(im.shape) < 3
    imgwidth = im.shape[len(im.shape)-2]
    imgheight = im.shape[len(im.shape)-1]
    if (i == -1 and j == -1):
        i = int(uniform(0, imgwidth-w-1))
        j = int(uniform(0, imgheight-h-1))
    if is_2d:
        im_patch = im[i:i+w, j:j+h]
    else:
        im_patch = im[:, i:i+w, j:j+h]
    return im_patch, i, j


def rotated_expansion(imgs):
    shape = [imgs.shape[i] for i in range(len(imgs.shape))]
    shape[0] = shape[0]*4 # there will be 4 times as many images after we rotate in each direction
    shape = tuple(shape)
    rotated_imgs = np.empty(shape)
    
    for index in range(int(shape[0]/4)):
        img = imgs[index]
        if(len(np.shape(img))>2):
            img90 = np.rot90(img.swapaxes(1,2).swapaxes(0,2)).swapaxes(0,2).swapaxes(1,2)
            img180 = np.rot90(img90.swapaxes(1,2).swapaxes(0,2)).swapaxes(0,2).swapaxes(1,2)
            img270 = np.rot90(img180.swapaxes(1,2).swapaxes(0,2)).swapaxes(0,2).swapaxes(1,2)
        else:
            img90 = np.rot90(img)
            img180 = np.rot90(img90)
            img270 = np.rot90(img180)
        
        rotated_imgs[index*4] = img
        rotated_imgs[index*4+1] = img90
        rotated_imgs[index*4+2] = img180
        rotated_imgs[index*4+3] = img270
    
    return rotated_imgs

def flipped_expansion(imgs):
    shape = [imgs.shape[i] for i in range(len(imgs.shape))]
    shape[0] = shape[0]*3 # there will be 4 times as many images after we rotate in each direction
    shape = tuple(shape)
    flipped_imgs = np.empty(shape)
    
    for index in range(int(shape[0]/3)):
        img = imgs[index]
        if(len(np.shape(img))>2):
            imgup = np.flipud(img.swapaxes(1,2).swapaxes(0,2)).swapaxes(0,2).swapaxes(1,2)
            imglr = np.fliplr(img.swapaxes(1,2).swapaxes(0,2)).swapaxes(0,2).swapaxes(1,2)
        else:
            imgup = np.flipud(img)
            imglr = np.fliplr(img)
        
        flipped_imgs[index*3] = img
        flipped_imgs[index*3+1] = imgup
        flipped_imgs[index*3+2] = imglr
    
    return flipped_imgs

def load_images_and_grounds():

	# Loading a set of 100 training images
	root_dir = "data/train/"

	image_dir = root_dir + "image/"
	files = os.listdir(image_dir)
	n = min(1000, len(files)) # Load maximum 1000 images
	imgs = np.array([load_image(image_dir + files[i]) for i in range(n)]).swapaxes(1,3).swapaxes(2,3)

	image_dir = root_dir + "label/"
	files = os.listdir(image_dir)
	n = min(1000, len(files)) # Load maximum 1000 images
	grounds = [load_image(image_dir + files[i]) for i in range(n)]

	print("Successfully loaded the training images and grounds.\n")

	imgs = np.array(imgs)
	grounds = np.array(grounds)

	return imgs, grounds


def crop_images_train(end_train, end_validation, imgs, grounds):

	# crop images to their 256*256 counterparts
	cropped_imgs = []
	cropped_targets = []

	for i in range(end_validation):
	    cropped_img, k, l = img_rnd_crop(imgs[i], 256, 256)
	    cropped_target, _, _ = img_rnd_crop(grounds[i], 256, 256, k, l)
	    cropped_imgs.append(cropped_img)
	    cropped_targets.append(cropped_target)

	x = list(range(end_validation))
	random.shuffle(x)

	train_input = [cropped_imgs[i] for i in x[:end_train]] #normally = 0:1080
	validation_input = [cropped_imgs[i] for i in x[end_train:end_validation]] #normally = 1080:1200



	train_target = [cropped_targets[i] for i in x[:end_train]] #normally = 0:1080
	validation_target = [cropped_targets[i] for i in x[end_train:end_validation]] #normally = 1080:1200

	return train_input, validation_input, train_target, validation_target



def load_test():
	root_dir = "data/"
	image_dir = root_dir + "test/"
	files = os.listdir(image_dir)
	n = min(1000, len(files)) # Load maximum 1000 images
	test_images = [load_image(image_dir + files[i]) for i in range(n)]
	test_images = np.array(test_images)

	return test_images


def uncrop_256_to_608(imgs):
    output_shape = (3,608,608)
    
    interval_tl = slice(0,256) #interval corresponding to left of x axis and top of y axis
    interval_c = slice(176,432) #interval corresponding to center of both axis
    interval_br = slice(352,608) #interval corresponding to right of x axis and bottom of y axis
    
    top_left = np.zeros(output_shape)
    top_center = np.zeros(output_shape)
    top_right = np.zeros(output_shape)
    center_left = np.zeros(output_shape)
    true_center = np.zeros(output_shape)
    center_right = np.zeros(output_shape)
    bottom_left = np.zeros(output_shape)
    bottom_center = np.zeros(output_shape)
    bottom_right = np.zeros(output_shape)
    
    top_left[:,interval_tl, interval_tl] = imgs[0]
    top_center[:,interval_tl, interval_c] = imgs[1]
    top_right[:,interval_tl, interval_br] = imgs[2]
    center_left[:,interval_c, interval_tl] = imgs[3]
    true_center[:,interval_c, interval_c] = imgs[4]
    center_right[:,interval_c, interval_br] = imgs[5]
    bottom_left[:,interval_br, interval_tl] = imgs[6]
    bottom_center[:,interval_br, interval_c] = imgs[7]
    bottom_right[:,interval_br, interval_br] = imgs[8]
    
    output = np.max([top_left, top_center, top_right, 
                     center_left, true_center, center_right,
                     bottom_left, bottom_center, bottom_right], axis=0).astype(np.uint8)
    return output
