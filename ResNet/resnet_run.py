# import PyTorch, Torchvision packages
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torchvision import transforms

# import Numpy
import numpy as np

# import Maplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

#import PIL
from PIL import Image

# import Python standard library
import time
import os,sys
import random

# import custom modules
from image_manipulation import *
from models import *
from submission_helper import *
from implementations import *

# Helper functions

def resnet_run(starting_model = "", is_segmented = False):

    
    # If installed, cuda allows to decrese computational time a lot
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Choosing the model

    if(starting_model == ""):
        if(not is_segmented):
            model, _, _ = train()
        else:
            model, _, _ = train(augmentation = "segmented")
    else:
            
        try:
            # Definition of model
            model = resnet_50(device, starting_model)
        except:
            print("Seems like the model is not present. Please download it as documented in README.md")
            raise SystemExit
    
    
    # Load test data set
    if (is_segmented):
        # Peculiar segmented data set
        
        test = np.array(np.load("data/npydata/seg_crop_test.npy").swapaxes(1,3).swapaxes(2,3))
        test_images=[]
        for i in range(int(np.shape(test)[0]/9)):
            test_images.append(uncrop_256_to_608(test[i*9:(i+1)*9]/255))

        test_images = np.array(test_images)

    else:
        test_images = load_test()
        
    
    # Run the test
    pred = run_test(test_images, model, device)

    # Creates the submission
    
    if os.path.exists("ResNet"):
        submission_filename = "ResNet/results/submission.csv"
    else:
        submission_filename = "results/submission.csv"
    masks_to_submission(submission_filename, *pred)


if __name__ == "__main__":
    """
    Vary the parameter in train to obtain submissions:

    starting_model (string) :   This parameters allows you to start the training with a pretrained model. Specify the
                                name of the file in the results folder, without specifying the name of the folder. If
                                you don't want a starting model, either don't specify this input, or write "".
                                Default value: ""

    is_segmented (bool) :       Specify whether you want to use the segemented data set.
                                Default value : False
    """

    resnet_run()
