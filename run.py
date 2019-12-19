import os
from augmentation import *

augment_dataset()

os.system('python ./ResNet/best_run.py')