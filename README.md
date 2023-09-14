# SAQSM
Source code for the paper “A spatially adaptive regularization constraint based three-dimensional reconstruction network for susceptibility imaging”

## A spatially adaptive regularization constraint based three-dimensional reconstruction network for susceptibility imaging
The implementation of SAQSM for QSM reconstruction network  <br />
This code was tested on Linux18.04

## Overall Framework


Fig.1 The architecture of our proposed SAQSM network. 

## Python Dependencies
numpy <br />
tensorflow_gpu-1.14.0 <br />
scipy <br />


# About this repository

- `/model.SAQSM` SAQSM source code
- `/model.SAQSM_3SAM` SAQSM with 3 SAMs
- `/model.SAQSM_2SAM` SAQSM with 2 SAMs
- `/model.SAQSM_1SAM` SAQSM with 1 SAMs
- `/model.SAQSM_0SAM` SAQSM with 0 SAMs
- `/model.SAQSM_NoRB` SAQSM without the residual block
- `/model.SAQSM_NoWeight` SAQSM without the channel weighting
- `/model.SAQSM_NoIB` SAQSM without the cascaded Inception blocks
- `/model.Unet_NoBN` U-Net without batch normalization
- `/SAQSM_train` Code for training SAQSM
- `/SAQSM_test` Code for testing SAQSM

