# import PyTorch, Torchvision packages
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torchvision import transforms


# import Numpy
import numpy as np

# Convolutions models

def resnet_50(device, best_model = "")

	res50_conv = torch.nn.Sequential(*list(
	        torchvision.models.resnet50(pretrained=True).children())[:-2])  # get all layers except avg-pool & fc


	# for p in res50_conv.parameters():
	#     p.requires_grad=False

	model = torch.nn.Sequential(
	    res50_conv,  # encoder
	    torch.nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),  # 2x upsample
	    torch.nn.BatchNorm2d(1024),
	    torch.nn.ReLU(),
	    torch.nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # 2x upsample
	    torch.nn.BatchNorm2d(512),
	    torch.nn.ReLU(),
	    torch.nn.ConvTranspose2d(512, 256, kernel_size=6, stride=4, padding=1),  # 4x upsample
	    torch.nn.BatchNorm2d(256),
	    torch.nn.ReLU(),
	    torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 2x upsample
	    torch.nn.BatchNorm2d(128),
	    torch.nn.ReLU(),
	    torch.nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0),  # logits per pixel
	    torch.nn.Sigmoid()  # predictions per pixel  # could remove and use BCEWithLogitsLoss instead of BCELoss.
	)
	model.to(device)



	for p in model.parameters():
	    try:
	        torch.nn.init.xavier_normal_(p)
	    except ValueError:
	        pass
	    
	if (best_model !=""):
		model = torch.load("results/"+best_model)

	return model

# Losses Models

class GrayscaleAndThreshold:
	    """ Reduce image to a single binary channel """
	    def __init__(self, level=0.1):
	        self.level = level

	    def __call__(self, img):
	        img = img.convert('L')  # 0..255, single channel

	        np_img = np.array(img, dtype=np.uint8)
	        np_img[np_img > self.level*255] = 255
	        np_img[np_img <= self.level*255] = 0

	        img = Image.fromarray(np_img, 'L')

	        return img

	class WeightedBCELoss(torch.nn.BCELoss):
	    def __init__(self, class_weights=None):  # does not support weight, size_average, reduce, reduction
	        super().__init__(reduction='none')
	        if class_weights is None:
	            class_weights = torch.ones(2)
	        self.class_weights = torch.as_tensor(class_weights)

	    def forward(self, input, target):
	        raw_loss = super().forward(input, target)
	        class_weights = self.class_weights.to(input.device)
	        weight_matrix = class_weights[0]*(1-target) + class_weights[1]*target
	        loss = weight_matrix * raw_loss
	        loss = loss.mean()  # reduction='elementwise_mean'
	        return loss


	def compute_class_weights(imgs):
	    mask_transform = transforms.Compose([
	        GrayscaleAndThreshold(),
	        transforms.ToTensor()
	        ])

	    road_pxs = 0
	    bg_pxs = 0
	    for img in imgs:
	        img = Image.fromarray(np.uint8(img*255))
	        mask_tr = torch.squeeze(mask_transform(img)).numpy().astype(int)
	        road_pxs += mask_tr.sum()
	        bg_pxs += (1 - mask_tr).sum()

	    bg_px_weight = (road_pxs + bg_pxs) / (2 * bg_pxs)  # "class 0"
	    road_px_weight = (road_pxs + bg_pxs) / (2 * road_pxs)  # "class 1"

	    return bg_px_weight, road_px_weight
