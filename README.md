# neural-image-assessment
Image Assessment using Neural Networks 
[![MIT License](https://img.shields.io/badge/MIT-License-brightgreen)](./LICENSE)

This is a PyTorch implementation of the paper [NIMA: Neural IMage Assessment](https://arxiv.org/abs/1709.05424) (accepted at [IEEE Transactions on Image Processing](https://ieeexplore.ieee.org/document/8352823)) by Hossein Talebi and Peyman Milanfar. You can learn more from [this post at Google Research Blog](https://research.googleblog.com/2017/12/introducing-nima-neural-image-assessment.html).

## Implementation Details
+ Dataset used for training: [AVA dataset](http://refbase.cvc.uab.es/files/MMP2012a.pdf) containing 255,500+ images. You can get it from [here](https://github.com/mtobeiyf/ava_downloader). ~~**Note: there may be some corrupted images in the dataset, remove them first before you start training**.~~ Use provided CSVs which have already done this for you.
