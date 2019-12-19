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

# Helper functions


def train(augmentation = "normal", full_data = False, starting_model = "", optimizer_name = "SGD", n_epochs = 20, do_save = True, seed = 42):

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Creating train and validation data
    if (augmentation == "rotated_flipped"):
        if(full_data):
            end_train = 1000
            end_validation = 1200
        else:
            end_train = 500
            end_validation = 550
        imgs, grounds = load_images_and_grounds()

        rotated_grounds = rotated_expansion(grounds)
        rotated_imgs = rotated_expansion(imgs)
        flipped_rotated_grounds = flipped_expansion(rotated_grounds)
        flipped_rotated_imgs = flipped_expansion(rotated_imgs)

        train_input, validation_input, train_target, validation_target = crop_images_train(end_train, end_validation, flipped_rotated_imgs, flipped_rotated_grounds)
    if (augmentation == "segmented"):
        # Particular segmented data set

        train = np.array(np.load("data/npydata/seg_crop_train.npy").swapaxes(1,3).swapaxes(2,3))
        target = np.array(np.load("data/npydata/seg_crop_target.npy"))
        x = list(range(400))
        random.shuffle(x)

        train_input = [train[i]/255 for i in x[:320]]
        validation_input = [train[i]/255 for i in x[320:400]]

        train_target = [target[i]/255 for i in x[:320]]
        validation_target = [target[i]/255 for i in x[320:400]]

    else:
        if (augmentation != "normal"):
            print("Augmentation to be done unclear, will be using normal augmentation.\n")
        end_train = 90
        end_validation = 100
        imgs, grounds = load_images_and_grounds()
        train_input, validation_input, train_target, validation_target = crop_images_train(end_train, end_validation, imgs, grounds)


    # If installed, cuda allows to decrese computational time a lot
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Definition of model
    model = resnet_50(device, starting_model)

    # Data statistics
    class_weights = compute_class_weights(train_target)
    loss_function = WeightedBCELoss(class_weights)

    # Choice of optimizer
    if (optimizer == "SGD"):
        optimizer = optim.SGD(model.parameters(), lr=3.75e-2, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)


    #start training
    loss_train_epoch =[]
    loss_validation_epoch =[]
    best_validation_loss = float("inf")
    best_epoch = -1

    #Time for printing
    training_start_time = time.time()

    #Loop for n_epochs
    for epoch in range(n_epochs):

        total_loss = 0.0

        for index in range(np.shape(train_input)[0]):
            model.train()

            input_image = torch.tensor(train_input[index]).unsqueeze(0).to(device)
            target_image = torch.tensor(train_target[index]).to(device)

            #Forward pass, backward pass, optimize
            outputs = model(input_image.float())
            loss = loss_function(outputs, target_image.float())

            #Set the parameter gradients to zero
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #Print statistics
            total_loss += loss.item() * input_image.size(0)

            print("Epoch", epoch, ", image", index, ", image loss:", loss.item(), ", time elapsed:", time.time() - training_start_time)

        #At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        for index in range(np.shape(validation_input)[0]):
            model.eval()


            input_image = torch.tensor(validation_input[index]).unsqueeze(0).to(device)
            target_image = torch.tensor(validation_target[index]).to(device)

            #Forward pass
            val_outputs = model(input_image.float())
            val_loss = loss_function(val_outputs, target_image.float())
            total_val_loss += val_loss.item() * input_image.size(0)

        print("Validation loss for epoch", epoch, ":", total_val_loss/np.shape(validation_input)[0])

        loss_train_epoch.append(total_loss/np.shape(train_input)[0])
        loss_validation_epoch.append(total_val_loss/np.shape(validation_input)[0])

        if(loss_validation_epoch[-1] < best_validation_loss):
            best_validation_loss = loss_validation_epoch[-1]
            best_epoch = epoch
            best_model = model
            if(do_save):
                torch.save(model, 'best_model.pth')

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

    if(do_save):
        torch.save(model, 'results/last_model.pth')
        np.savetxt('results/losstrain.csv', loss_train_epoch, delimiter=',')
        np.savetxt('results/loss_validation.csv', loss_validation_epoch, delimiter=',')



    return best_model, best_validation_loss, best_epoch


def run_test(test_images, model, device):
    crop = 256
    masks = []
    
    for test_image in test_images:
        imgheight = test_image.shape[1]
        imgwidth = test_image.shape[2]
        mask = torch.zeros(1, imgheight, imgwidth)
        for i in range(0, imgheight, crop):
            for j in range(0, imgwidth, crop):
                # when the crop is bigger than the image size, we increase the temporary image with 0
                if(i+crop>imgheight and j+crop>imgwidth):
                    im_patch = np.zeros([3,crop,crop],dtype = np.float32)
                    im_patch[:, :imgheight-i, :imgwidth-j] = test_image[:, i:imgheight, j:imgwidth]
                    im_patch = torch.tensor(im_patch).unsqueeze(0).to(device)
                    mask[:, i:imgheight, j:imgwidth] = model(im_patch.float()).detach()[0,0,:imgheight-i,:imgwidth-j]

                elif(i+crop>imgheight):
                    im_patch = np.zeros([3,crop,crop],dtype = np.float32)
                    im_patch[:, :imgheight-i, :] = test_image[:, i:imgheight, j:j+crop]
                    im_patch = torch.tensor(im_patch).unsqueeze(0).to(device)
                    mask[:, i:imgheight, j:j+crop] = model(im_patch.float()).detach()[0,0,:imgheight-i,:]

                elif(j+crop>imgwidth):
                    im_patch = np.zeros([3,crop,crop])
                    im_patch[:, :, :imgwidth-j] = test_image[:, i:i+crop, j:imgwidth]
                    im_patch = torch.tensor(im_patch).unsqueeze(0).to(device)
                    mask[:, i:i+crop, j:imgwidth] = model(im_patch.float()).detach()[0,0,:,:imgwidth-j]

                else: # cas normal
                    im_patch = test_image[:, i:i+crop, j:j+crop]
                    im_patch = torch.tensor(im_patch).unsqueeze(0).to(device)
                    mask[:, i:i+crop, j:j+crop] = model(im_patch.float()).detach()[0,0,:,:]
        masks.append(mask.numpy())
    return masks


if __name__ == '__main__':
    """
    Vary the parameter in train to obtain different model parameters:

    augmentation (string) :     Can be set to either "normal", "rotated_flipped" or "segmented".
                                This parameter sets different parameters conditions for the input data. 
                                The first one makes no augmentations, the second one augments the data 
                                set with rotations and flipping and the third one uses a special segmented
                                data set.
                                Default value = "normal"

    full_data (bool) :          This parameter set wether you want to use the full data for the "rotated_flipped"
                                data set. If set to "True" the full 1200 data of the original data set will be used,
                                if set to "False" only 550 data will be used. Consider using "False" since it halves
                                the already long computational time and it tends to overfits.
                                Default value = "False"

    starting_model (string) :   This parameters allows you to start the training with a pretrained model. Specify the
                                name of the file in the results folder, without specifying the name of the folder. If
                                you don't want a starting model, either don't specify this input, or write "".
                                Default value: ""

    optimizer_name (string) :   This specifies the optimizer to use between SGD and Adam.
                                Default value: "SGD"

    n_epochs (int) :            Specify the number of epoch to be done.
                                Default value : 20

    do_save (bool) :            Specify whether you want to save the best model, last model and the losses values in
                                the results folder.
                                Default value : True

    seed (int) :				Sets a value for the random seed.
                                Default value : 42
    """

    _, _, _ = train()