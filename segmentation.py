import numpy as np
from skimage.color import rgb2gray
from cropping import *

#functions
def apply_threshold(img, threshold):
    output = (255*(np.sign(img - threshold*np.ones(img.shape)) + 1)/2).astype('uint8')
    return output

def segment_dataset(dataset, unet_path, resnet_path):
    thresholds = [0.3, 0.35, 0.4]
    threshold_set = np.empty(dataset.shape, dtype='uint8')

    for i in range(len(dataset)):
        img = rgb2gray(dataset[i])

        for j in range(len(thresholds)):
            threshold_set[i,:,:,j] = apply_threshold(img, thresholds[j])
        
    #save to resnet and unet
    np.save(resnet_path, threshold_set)
    png_dataset_to_tif(threshold_set, unet_path)
############################################

if __name__ == "__main__":
    # load cropped dataset
    train_crop, target_crop, test_crop = crop_dataset(False)

    #segment train and test dataset
    os.mkdir('ResNet/data/npydata/')
    segment_dataset(train_crop, 'UNET/data/train/image', 'ResNet/data/npydata/seg_crop_train.npy')
    segment_dataset(test_crop, 'UNET/data/test', 'ResNet/data/npydata/seg_crop_test.npy')

    #save groundtruths, since they don't have to be segmented
    png_dataset_to_tif(target_crop, './UNET/data/train/label')
    np.save('ResNet/data/npydata/seg_crop_target.npy', target_crop)