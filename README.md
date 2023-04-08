
# Transformer implementation  for CIFAR10

This repository contains the scratch implementation of a simple **Transformer** architecture for the CIFAR10 dataset.

## Data

The CIFAR10 dataset was imported using `torchvision.datasets` and augmented using the [albumentations](https://albumentations.ai) library. The following augmentations were applied on the data:
* RandomCrop of 32, 32 (after padding of 4)
* FlipLR 
* CutOut(8, 8)


## Architecture


*  3 Convolutions to arrive at AxAx48 dimensions (e.g. 32x32x3 | 3x3x3x16 >> 3x3x16x32 >> 3x3x32x48)
* Apply GAP and get 1x1x48, call this X
* Create a block called ULTIMUS that:
    * Creates 3 FC layers called K, Q and V such that:
        * X*K = 48*48x8 > 8
        * X*Q = 48*48x8 > 8 
        * X*V = 48*48x8 > 8 
    * then create AM = SoftMax(Q.T * K)/(8^0.5) = 8*8 = 8
    * then Z = V*AM = 8*8 > 8
    * then another FC layer called Out such that: 
        * Z*Out = 8*8x48 > 48
* Repeat this Ultimus block 4 times
* Then add final FC layer that converts 48 to 10 and sends it to the loss function.



