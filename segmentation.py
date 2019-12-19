import numpy as np
from skimage.color import rgb2gray
from cropping import *

#functions
def apply_thresholds(img, thresholds):
    output = np.zeros(img.shape)
    
    for threshold in thresholds:
        output += threshold*(np.sign(img - threshold*np.ones(img.shape)) + 1)/2
        
    return output

def segment_dataset(dataset, unet_path, resnet_path):
    thresholds = [0.3, 0.35, 0.4]
    threshold_train_set = np.empty(dataset.shape)

    for i in range(len(dataset)):
        img = rgb2gray(dataset[i])

        for j in range(len(thresholds)):
            threshold_set[i,:,:,j] = apply_threshold(img, thresholds[j])
        
    #save to resnet and unet
    np.save(resnet_path, threshold_set)
    png_dataset_to_tif(threshold_set, unet_path)
############################################

    
# load cropped dataset    
train_crop, target_crop, test_crop = crop_dataset(save=False)

#segment train and test dataset
segment_dataset(train_crop, 'UNET/data/train/image', 'ResNet/data/npydata/seg_crop_train.npy')
segment_dataset(test_crop, 'UNET/data/test', 'ResNet/data/npydata/seg_crop_test.npy')

#save groundtruths, since they don't have to be segmented
png_dataset_to_tif(target_crop, './UNET/data/train/label')
np.save('ResNet/data/npydata/seg_crop_target.npy')