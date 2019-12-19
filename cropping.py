import numpy as np
import os, sys
import shutil
import matplotlib.image as mpimg
from skimage import io

# helper functions
def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

def png_dataset_to_tif(dataset, output_path):
    
    for i in range(len(dataset)):
        filename='/' + str(i) + '.tif'
        io.imsave(output_path + filename, dataset[i])
        
# takes a 400 by 400 image as input and returns 4 256x256 images as output
def crop_400_to_256(img):
    top_left = img[0:256, 0:256]
    top_right = img[0:256, 144:400]
    bottom_left = img[144:400, 0:256]
    bottom_right = img[144:400, 144:400]
    
    return [top_left, top_right, bottom_left, bottom_right]

# takes a 608x608 image as input and returns 9 256x256 images as output
def crop_608_to_256(img):
    interval_tl = slice(0,256) #interval corresponding to left of x axis and top of y axis
    interval_c = slice(176,432) #interval corresponding to center of both axis
    interval_br = slice(352,608) #interval corresponding to right of x axis and bottom of y axis
    
    top_left = img[interval_tl, interval_tl]
    top_center = img[interval_tl, interval_c]
    top_right = img[interval_tl, interval_br]
    center_left = img[interval_c, interval_tl]
    true_center = img[interval_c, interval_c]
    center_right = img[interval_c, interval_br]
    bottom_left = img[interval_br, interval_tl]
    bottom_center = img[interval_br, interval_c]
    bottom_right = img[interval_br, interval_br]
    
    return [top_left, top_center, top_right, center_left, true_center, center_right, bottom_left, bottom_center, bottom_right]
##########################################################

def crop_dataset(save):
    # Load the dataset
    root_dir = "./dataset/training/"

    image_dir = root_dir + "images/"
    try:
        files = os.listdir(image_dir)
    except:
        print("Seems like the dataset is not present, please download it and put it in the /dataset/ folder as documented in README.md")
        raise SystemExit
        
    n = min(100, len(files)) # Load maximum 100 images
    imgs = img_float_to_uint8([load_image(image_dir + files[i]) for i in range(n)])
        

    image_dir = root_dir + "groundtruth/"
    files = os.listdir(image_dir)
    n = min(100, len(files)) # Load maximum 100 images
    gts = img_float_to_uint8([load_image(image_dir + files[i]) for i in range(n)])

    root_dir = "./dataset/test_set_images/"
    test_images=[]
    for i in range(1, 51):
        image_filename = root_dir + "test_" + str(i) + "/test_" + str(i) + '.png'
        test_images.append(np.array(load_image(image_filename)))
    test_imgs = img_float_to_uint8(test_images)


    # crop train data
    train_shape = list(imgs.shape)
    train_crop_shape = (4*train_shape[0], 256, 256, 3)
    train_input_crop = np.empty(train_crop_shape).astype(np.uint8)

    for index in range(len(imgs)):
        image = imgs[index]
        train_input_crop[4*index:4*index+4] = np.asarray(crop_400_to_256(image))

    # crop groundtruths
    target_shape = list(gts.shape)
    target_crop_shape = (4*target_shape[0], 256, 256)
    target_crop = np.empty(target_crop_shape).astype(np.uint8)

    for index in range(len(gts)):
        image = gts[index]
        target_crop[4*index:4*index+4] = np.asarray(crop_400_to_256(image))

    #crop test data       
    test_shape = list(test_imgs.shape)
    test_crop_shape = (9*test_shape[0], 256, 256, 3)
    test_crop = np.empty(test_crop_shape).astype(np.uint8)

    for index in range(len(test_imgs)):
        image = test_imgs[index]
        test_crop[9*index:9*index+9] = np.asarray(crop_608_to_256(image))

    # save cropped data to unet
    if save:
        png_dataset_to_tif(train_input_crop, './UNET/data/train/image')
        png_dataset_to_tif(target_crop, './UNET/data/train/label')
        png_dataset_to_tif(test_crop, './UNET/data/test')
        
        # save data to ResNet. Saved data is not cropped because ResNet will crop the data itself
        path = './ResNet/data'
        if os.path.exists(path):
            shutil.rmtree(path) # remove folder because copytree destination must not exist
        shutil.copytree('./dataset/', path)
    
    return train_input_crop, target_crop, test_crop

if __name__ == "__main__":
    crop_dataset(True)