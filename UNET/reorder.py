#This function is needed because the implementation of unet.py doesn't take the inputs in the correct order.
#This function reorders the outputs according to the inputs so that 0.tif in the results corresponds to 0.tif in the inputs

import numpy as np
import glob
import os
from keras.preprocessing.image import array_to_img

def reorder_results(path_to_test_npy, path_to_test_intput_folder, path_to_output_folder):
    results = np.load(path_to_test_npy)
    imgnames = glob.glob(path_to_test_intput_folder + "*.tif")
    
    #finding the desired order
    indices = []
    for imgname in imgnames:
        indices.append(int(imgname[imgname.rindex("\\")+1:imgname.rindex(".")]))
    print(len(indices))
    
    #reordering
    ordered_results = np.empty(results.shape)
    for i in range(len(results)):
        image = results[i]
        index = indices[i]

        ordered_results[index] = image
    
    #saving
    for i in range(ordered_results.shape[0]):
        img = ordered_results[i]
        img = array_to_img(img)
        img.save(path_to_output_folder + "/%d.jpg"%(i))
    np.save(path_to_output_folder + "ordered_results.npy", ordered_results)