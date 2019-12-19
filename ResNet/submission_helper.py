#import numpy
import numpy as np

#import PIL
from PIL import Image

# import Python standard library
import time
import os,sys
import random

# assign a label to a patch
def patch_to_label(patch):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def mask_to_submission_strings(image, img_number):
    """Reads a single image and outputs the strings that should go into the submission file"""
    patch_size = 16
    for j in range(0, image.shape[1], patch_size):
        for i in range(0, image.shape[2], patch_size):
            patch = image[0][i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))

def masks_to_submission(submission_filename, *images):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        i=int(0)
        for image in images[0:]:
            i+=1
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(image,i))