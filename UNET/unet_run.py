from data import *
from unet import *
from reorder import *
from cropping import *
from submission_helper import *

# data processing for unet ##################
mydata = dataProcess(256,256)
mydata.create_train_data()
mydata.create_test_data()

# train and test net ########################
myunet = myUnet()
myunet.train()
myunet.save_img()

#since the outputs are not in the same order as the inputs, reorder
reorder_results("./results/imgs_mask_test.npy", "./data/test/", "./results/")

#since the outputs are cropped, uncrop
uncrop_results("./results/ordered_results.npy")

#create submission
to_upload_mask = np.reshape(np.load('./results/uncropped_output_mask.npy'), (50,1,608,608))
masks_to_submission('./submission_base.csv', *to_upload_mask)