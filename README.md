By Susan Potters and Sjoerd Dijkstra
Link to code: https://github.com/SusanPotters/DL_PSMNet-

# Introduction
As a part of the course CS4240 Deep Learning it was mandated that a project is conducted. The project was to consist of a reproducibility of one of a pre-determined list of papers on deep learning. One such paper was that  of `Pyramid Stereo Matching Network' by Jia-Ren Chang and Yong-Sheng Chen. We chose to conduct this reproduction on the results as obtained by the paper and  extend work towards the use of such a result on an external dataset for application. 
In this work the writers proposed a new pyramid stereo matching network to be further referred to as PSMNet. This network aimed to exploit global context information in stereo matching. In order to make it possible that the receptive field could be enlarged, spatial pyramid pooling and dilated convolution were used. By doing so, the network is capable of extending pixel-level feature maps to region-level feature maps with varying scales to receptive fields. Disparity estimation was conducted reliably using the combination of these global and local feature maps to form a cost volume. A stacked hourglass 3D CNN in conjunction with intermediate supervision was designed to regularize this cost volume.  This architecture processed the cost volume in a top-down/bottom-up manner to improve on the existing use of global context information.

Their main contributions are summarised as follows:

1. End-to-end learning framework for stereo matching without post-processing
2. Pyramid pooling module for utilising global context information into image features
3. Stacked hourglass 3D CNN as an extension to the regional support of context information in cost volume
4. Achievement of state-of-the-art accuracy on the KITTI dataset

# Pyramid Stereo Matching Network
