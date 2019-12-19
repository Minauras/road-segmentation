import numpy as np
import random
from cropping import *

#this function augments the dataset by flipping and rotating
def flip_rot_expansion(imgs):
    return flip_expansion(rotated_expansion(imgs))

#this function augments the dataset by rotating all pictures in all directions
def rotated_expansion(imgs):
    imgs0 = imgs
    imgs90 = np.empty(imgs0.shape)
    imgs180 = np.empty(imgs0.shape)
    imgs270 = np.empty(imgs0.shape)
    
    for index in range(len(imgs)):
        img = imgs[index]
        img90 = np.rot90(img)
        img180 = np.rot90(img90)
        img270 = np.rot90(img180)
        
        imgs90[index] = img90
        imgs180[index] = img180
        imgs270[index] = img270
    
    rotated_imgs = np.append(imgs0, imgs90)
    rotated_imgs = np.append(rotated_imgs, imgs180)
    rotated_imgs = np.append(rotated_imgs, imgs270)
    
    return rotated_imgs

#this function augments the dataset by flipping all pictures horizontally and vertically
def flip_expansion(imgs):
    imgs_or = imgs
    imgs_lr = np.empty(imgs_or.shape)
    imgs_ud = np.empty(imgs_or.shape)
    
    for index in range(len(imgs)):
        img = imgs[index]
        
        img_lr = np.fliplr(img)
        img_ud = np.flipud(img)
        
        imgs_lr[3*index+1] = img_lr
        imgs_ud[3*index+2] = img_ud
    
    flipped_imgs = np.append(imgs_or, imgs_lr)
    flipped_imgs = np.append(flipped_imgs, imgs_ud)
    return flipped_imgs

def select_random(train, target, size):
    indices = [i for i in range(len(imgs))]
    random_indices = random.sample(indices, size)
    
    selected_train = train[random_indices]
    selected_target = target[random_indices]
    
    return selected_train, selected_target

# load cropped dataset    
train_crop, target_crop, test_crop = crop_dataset(save=False)

# augment training dataset
train_aug = flip_rot_expansion(train_crop)
target_aug = flip_rot_expansion(target_crop)

#select 400 samples
selected_train, selected_target = select_random(train_aug, target_aug, 400)

# save data to unet
png_dataset_to_tif(selected_train, './UNET/data/train/image')
png_dataset_to_tif(selected_target, './UNET/data/train/label')
png_dataset_to_tif(test_crop, './UNET/data/test') #test is not augmented or randomly selected

# save data to ResNet. Saved data is not augmented because ResNet will augment the data itself
copy_tree('./dataset/', './ResNet/data/')