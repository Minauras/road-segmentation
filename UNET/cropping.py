import numpy as np

# takes a 400 by 400 image as input and returns 4 256x256 images as output
def crop_400_to_256(img):
    top_left = img[0:256, 0:256]
    top_right = img[0:256, 144:400]
    bottom_left = img[144:400, 0:256]
    bottom_right = img[144:400, 144:400]
    
    return [top_left, top_right, bottom_left, bottom_right]

# takes 4 256x256 black/white images as input and returns a combined 400x400 image of those
# the ordering of the 256x256 images must be topleft, topright, bottomleft, bottomright
def uncrop_256_to_400(imgs):
    output_shape = (400,400)
    
    top_left = np.zeros(output_shape)
    top_right = np.zeros(output_shape)
    bottom_left = np.zeros(output_shape)
    bottom_right = np.zeros(output_shape)
    
    top_left[0:256, 0:256] = imgs[0]
    top_right[0:256, 144:400] = imgs[1]
    bottom_left[144:400, 0:256] = imgs[2]
    bottom_right[144:400, 144:400] = imgs[3]
    
    output = np.max([top_left, top_right, bottom_left, bottom_right], axis=0).astype(np.uint8)
    return output

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

# takes 9 cropped 256x256 images and returns a combined 608x608 image of those
def uncrop_256_to_608(imgs):
    output_shape = (608,608)
    
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
    
    top_left[interval_tl, interval_tl] = imgs[0]
    top_center[interval_tl, interval_c] = imgs[1]
    top_right[interval_tl, interval_br] = imgs[2]
    center_left[interval_c, interval_tl] = imgs[3]
    true_center[interval_c, interval_c] = imgs[4]
    center_right[interval_c, interval_br] = imgs[5]
    bottom_left[interval_br, interval_tl] = imgs[6]
    bottom_center[interval_br, interval_c] = imgs[7]
    bottom_right[interval_br, interval_br] = imgs[8]

    
    
    output = np.max([top_left, top_center, top_right, 
                     center_left, true_center, center_right,
                     bottom_left, bottom_center, bottom_right], axis=0).astype(np.uint8)
    return output

#This is a specific function that uncrops the results of unet given a path to the result maks
def uncrop_results(path_to_results_npy):
    results = np.load(path_to_results_npy)
    results = (255*results).astype('uint8')
    
    nb_outputs_imgs = int(results.shape[0]/9)
    indices = [i for i in range(nb_outputs_imgs)]
    uncropped_output = np.empty((nb_outputs_imgs, 608, 608))
    for index in indices:    
        cropped_imgs = [results[9*index + i].squeeze(2) for i in range(9)]
        uncropped_output[index] = uncrop_256_to_608(cropped_imgs)
    
    np.save('./results/uncropped_output_mask.npy', uncropped_output)
    return uncropped_output