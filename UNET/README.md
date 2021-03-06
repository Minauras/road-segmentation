# Road segmentation using UNET

Here is the implementation of UNET for our road segmentation project, we will compare its performances with ResNet.

## Getting Started

Clone the master repository to your machine in order to use the implementation of UNET.

### Prerequisites

* Tensorflow
* Keras 2.0.0
* openCV 3

### Using your dataset

This implementation of unet runs on 256x256 images. If your images are not of this size, you need to crop them.
Functions to crop our road identification dataset can be found in file cropping.py.

* Put your training images in the folder data/train/image/ as .tif files named "0.tif" "1.tif" etc. They must be 256x256 3-channel (RGB) tif images.
* Put your training labels in the folder data/train/label/ as .tif files named "0.tif" "1.tif" etc. They must be 256x256 1-channel (grayscale) tif images. The image must have a 0 where you have class 0 in the input image, and 1 where you have class 1. The name of the label file must be the same as the name of the corresponding image file.
* Put your test images in the folder data/tes as .tif files named "0.tif" "1.tif" etc. They must be 256x256 3-channel (RGB) tif images.
* Leave the folder data/npydata empty. It will contain the masks after data is generated by the network
* Leave the folder results/ empty. It will contain the output images of the test and the mask and a csv file of submission format for submission on the aicrowd challenge.

## Running the Net

Once your data is put in the correct place, open a terminal and run:

```
python unet-run.py
```

This will generate the data masks, run the training of the model, run the testing of the model, and format the output data in the results folder.


## Authors

* **Rémi Clerc**  - [Minauras](https://github.com/Minauras)
* **Jonas Morin** - [Jono711](https://github.com/Jono711)
* **Jordan Metz** - [metzj](https://github.com/metzj)

Original work based on the [UNET repository](https://github.com/zhixuhao/unet) made by [zhixuhao](https://github.com/zhixuhao) (see [original_readme.md](original_readme.md)), and on [Paul Suskriti's tutorial](https://medium.com/coinmonks/learn-how-to-train-u-net-on-your-dataset-8e3f89fbd623).


## Acknowledgments

* We would like to thank zhixuhao for their work on UNET and Paul Suskriti for their very clear tutorial about zhixuhao's UNET.