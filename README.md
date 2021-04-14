By Susan Potters and Sjoerd Dijkstra.
[Link to code](https://github.com/SusanPotters/DL_PSMNet-).

# Introduction
As a part of the course CS4240 Deep Learning it was mandated that a project is conducted. The project was to consist of a reproducibility of one of a pre-determined list of papers on deep learning. One such paper was that  of `Pyramid Stereo Matching Network' by Jia-Ren Chang and Yong-Sheng Chen [1]. We chose to conduct this reproduction on the results as obtained by the paper and  extend work towards the use of such a result on an external dataset for application. 
In this work the writers proposed a new pyramid stereo matching network to be further referred to as PSMNet. This network aimed to exploit global context information in stereo matching. In order to make it possible that the receptive field could be enlarged, spatial pyramid pooling and dilated convolution were used. By doing so, the network is capable of extending pixel-level feature maps to region-level feature maps with varying scales to receptive fields. Disparity estimation was conducted reliably using the combination of these global and local feature maps to form a cost volume. A stacked hourglass 3D CNN in conjunction with intermediate supervision was designed to regularize this cost volume.  This architecture processed the cost volume in a top-down/bottom-up manner to improve on the existing use of global context information.

Their main contributions are summarised as follows:

1. End-to-end learning framework for stereo matching without post-processing
2. Pyramid pooling module for utilising global context information into image features
3. Stacked hourglass 3D CNN as an extension to the regional support of context information in cost volume
4. Achievement of state-of-the-art accuracy on the KITTI dataset

# Pyramid Stereo Matching Network
In order to get a better understanding of the significance of the achieved results we discuss the architecture of the model in a summarised manner. The full structure of the network can be seen in below image.

![DLRP1](https://user-images.githubusercontent.com/61514183/114722977-8f228400-9d3a-11eb-97b0-8ba696000c3a.png)

The first main change was made towards the standard large filters (7 x 7) design for the first convolution layer as done by other studies. Instead the writers opted for three smaller convolution layers (3 x 3) to be cascaded to construct a deeper network, however, with the same receptive field.
The second change was made towards the four basic residual blocks for learning unary feature extraction. Two of these basic residual blocks were transformed to conduct dilated convolution in order to enlarge the receptive field.
Then the model consisted of using an output feature map of a quarter of the size of the original input imaging and thus a sixteenth of the area. This input is then sent through the spatial pyramid pooling module (SPP) for context information. The feature maps of the left and right images are then concatenated into a cost volume. Using this cost volume, 3D CNN regularisation is performed to finally use regression to calculate the output disparity map.

## Spatial Pyramid Spooling Module
Just using the pixel intensities of images results in a loss of context relationships. As such an spatial pyramid pooling module is used to learn and thus use the relationships between the main regions that are objects and the sub-regions of that object in a hierarchical manner. One can think of the relationship between a car as the object and its wheels as a potential sub-region for that object. 
In the model provided four fixed-size average pooling blocks were used for SPP as can be seen from Figure 1. Furthermore, 1 x 1 convolution and upsampling are performed as done in previous works [2]. 

## Cost Volume
For the cost volume a similar approach as GC-Net is performed to concatenate the left and right features to learn matching cost estimation using a deep network. They follow the approach as done by [2]. For each stereo image a cost volume is represented in four dimensions: height, width, max disparity + 1 and feature size. This is calculated through concatenation of unary features through their left and right stereo image representations across each disparity level. 

## 3D CNN
The use of a custom stacked hourglass architecture allows the model to learn more context information through the repeated top-down/bottom-up processing with the presence of intermediate supervision. The architecture consists of three of these hourglass structures each of which produce their own disparity map with corresponding losses. During the training phase, the total loss is represented by the weighted sum of these losses. During testing, the final disparity map is the last of the three outputs. 

## Disparity Regression

To be able to be fully differentiable and regress a smooth disparity estimate, the paper makes use of disparity regression from [2]. This regression converts the predicted costs for each disparity from the cost volume to a probability volume by means of a softmax operation. 

# Models
The code for this project can be found at: https://github.com/SusanPotters/DL_PSMNet-.

## Pre-trained Model
The writers of the paper and makers of the model in the paper, provided a number of models that were pre-trained. The idea behind these models was that they could be used to prove their results and save those interested in the application of their models time as the model was trained using high-end graphical computing for a long period of time. The model we were most interested in was one of the models that was trained on the Scene Flow dataset over a run consisting of 10 epochs [3]. Further references to the pre-trained model refer to the model trained on the Scene Flow dataset.

## Self-trained Models
In addition to the pre-trained model,  we wanted to compare the results to that of a model that we ourselves had trained using their model as to prove reproducibility. We trained several models using the Scene Flow dataset. 
The first model consisted of training on the full Scene Flow training set. Subsets of this Scene Flow dataset are the FlyingThings3d, Driving and Monkaa datasets.  We trained this model to compare the papers baseline to their provided pre-trained model and our second model.
The second model consisted of training on only the Monkaa subset of the Scene Flow dataset. With this model we could get an idea as to the performance of the model on lesser amounts of data.
As a side note, we mention that the authors, during their training, had access to far superior hardware. The authors made use of four 12 GB GPUs in parallel. Training this model locally thus proved an impossible task as hardware made for consumer use could not handle even the slowest and potentially most inaccurate training procedures, using a batch size of one. As such we made use of a Google Cloud virtual machine to set up training. The virtual machine made use of a Nvidia V100 GPU. As we were only able to obtain one GPU, a batch size of 5 was used to train, whereas the authors used a batch size of 12. All other hyperparameters were left the same as described in the paper. The model trained on the full Scene Flow dataset was trained for only 5 epochs, as training was quite costly and the budget small. The model trained on only the Monkaa dataset was trained for 10 epochs, as was originally the plan. Moreover, doing ablation studies or tuning hyperparameters was not possible, due to the limited budget.

## External Data
Our external superviser wished to know whether or not a model of this kind could be used on data that it had not been trained on. The goal was to see if the model is able to make reasonable disparity imagery based on this data.
The data provided consisted of four images of motor blades. The photos were taken through a prisma and as such we were provided with left and right images in one photo. The data consisted of four such images, as can be seen below.

![Capture](https://user-images.githubusercontent.com/61514183/114724975-63a09900-9d3c-11eb-9c45-1afda7178dbb.PNG)

In order for this data to be used in the model, we first had to split the photos into equal sized left and right images. An example of how this was done is shown. These were then put through the model as left and right images respectively.

![leftright](https://user-images.githubusercontent.com/61514183/114725135-8a5ecf80-9d3c-11eb-91e4-634da02dbeb6.PNG)

# Results

## Results from the paper
First we provide the table of the main results of the paper using their trained model.  The authors of PSMNet obtained an EPE of 1.09 on the SceneFlow test set, which was significantly lower than earlier reported results at thetime the paper was published.

![DLRP7](https://user-images.githubusercontent.com/61514183/114725360-b8441400-9d3c-11eb-9425-9384cffb30d0.png)

## Pre-trained Model
![pretrainedres](https://user-images.githubusercontent.com/61514183/114725559-df024a80-9d3c-11eb-9153-27e1166eccd3.PNG)

The average EPE of the pre-trained model on the Scene Flow test set was 6.263, which is significantly higher than results reported by the authors. Multiple issues were opened on Github from fellow reproducers, who also found out that the results don't match [4,5] . The authors of the paper have therefore put a notice on their Github page that the output disparity is better when it is multiplied with a constant factor of 1.17. Indeed after doing this the EPE is reduced to 1.371. However, this is still above the reported 1.09 from Table 5 and higher than the second best model from Table 5 (CRL) [2]. Above figure shows error distributions for results of the trained model and results that were multiplied with 1.17 on the Scene Flow test set. As no reason was found for multiplying the results with 1.17, the original pre-trained model is further used for testing.

## Self-trained Models
### Scene Flow Model

![sceneflowerror](https://user-images.githubusercontent.com/61514183/114725960-3d2f2d80-9d3d-11eb-9f27-75f83f5a0797.PNG)

The average EPE of the self-trained Scene Flow model on the Scene Flow test set was 1.293. In above figure the distribution of the EPE for the Scene Flow test set is shown, as well as the training loss per epoch.

### Monkaa Model

![Monkaaerror](https://user-images.githubusercontent.com/61514183/114726097-589a3880-9d3d-11eb-9380-d24fb7eef315.PNG)

The average EPE of the self-trained Monkaa model on the Scene Flow test set was 3.213. Above figure shows the distribution of the EPE for the Scene Flow test set, as well as the training loss per epoch.

## Comparison Disparity Image Outputs

In this section we provide the reader with multiple images of the disparity outputs and EPE for the three models, as well as the actual disparity image.

![first](https://user-images.githubusercontent.com/61514183/114728030-03f7bd00-9d3f-11eb-95dc-9a5ad2afa8f2.PNG)
![second](https://user-images.githubusercontent.com/61514183/114728039-0528ea00-9d3f-11eb-9889-9c3594ea862e.PNG)
![third](https://user-images.githubusercontent.com/61514183/114728047-0823da80-9d3f-11eb-9602-88f30da7e5b9.PNG)

It can be seen that the self-trained Scene Flow model performs best for all given examples, obtaining an EPE below 1.0 for all cases. The pre-trained and Monkaa model obtain a relatively similar EPE. In addition, we can identify what points the models struggle with. For example, the Monkaa model seems to have difficulty with recognizing detailed shapes of objects, but predicts correct disparity values mostly for the objects that it does recognize. The pre-trained model is better at recognizing shapes of detailed objects, but does not predict correct disparity values. Lastly, the Scene Flow model is able to discern the shapes of objects and predict disparity values well. There are minimal differences when comparing it with the true disparity image.

## Test on External Dataset
All three models discussed above were used to test on the external dataset that was received by our external supervisor. These did not come with any information regarding the true disparity, so we can only speculate how well the results are based on the disparity images.

![DLRP9](https://user-images.githubusercontent.com/61514183/114726848-03125b80-9d3e-11eb-83e5-778385f12674.png)
![DLRP10](https://user-images.githubusercontent.com/61514183/114726858-04dc1f00-9d3e-11eb-84a6-b4d8323faba1.png)
![DLRP11](https://user-images.githubusercontent.com/61514183/114726860-06a5e280-9d3e-11eb-88dd-6552c306cba8.png)
![DLRP12](https://user-images.githubusercontent.com/61514183/114726868-07d70f80-9d3e-11eb-98ee-5003fd108d57.png)

Above figures show results for all four provided images. The first image is the original photo, the second contains results from the Monkaa model, the third contains results from the pre-trained model and the last shows results for the self-trained Scene Flow model. It can be seen that for all three models the output is not accurate and no grounded claims can be made which one performs best. Differences between the disparity of the background and the motor blade are not clear and there is a lot of noise. Moreover, the models are not able to handle shadow or different light conditions. We can think of multiple reasons why the model cannot generalize well to the motor blade images, as there are multiple inconsistencies between the training data and this test data. First, the motor blade images were not made with a stereo camera, but with a prism. Second, the provided photos were of a real indoor environment, while the training data consisted of synthetic images. 

## Test on InStereo2K Dataset
To evaluate if the model can generalize to real indoor images in general, a few tests were done on the InStereo2K dataset, which contains 2000 natural stereo images of indoor environments [7]. Results for two images from the self-trained Scene Flow model are shown below, where the upper images show the ground truth, the middle pictures show the results and the lowest pictures show the original image. They show that the model is not able to generalize to these natural indoor images as well. The shapes of objects are roughly recognized, but the disparity values are inaccurate and the model cannot cope with shadow. We think that the main reason for the inaccuracy is that the model was not trained on natural, but synthetic images. The creators of InStereo2K have used their dataset to finetune the PSMNet model trained on Scene Flow and report that they observe a better generalization performance, taking a step towards practical applications [7]. 

![together](https://user-images.githubusercontent.com/61514183/114727293-613f3e80-9d3e-11eb-97cb-9c4d29b58254.PNG)

# Conclusion
With regards to the reproducibility of the paper and provided materials, we can see a clear discrepancy. In Table 1 we note a EPE for the PSMNet of 1.09. However, the EPE acquired by the pre-trained and provided PSMNet model achieves an EPE of 6.263. As a result the model provided by the authors would not give a realistic result with regards to our desired output for the external data. 
On the other hand, our self-trained models, after only half of the epochs of the model as presumable trained in the paper, achieves and EPE of 1.293 for the Scene Flow model and an EPE of 3.213 for the Monkaa model. Both of these models thus outperform the provided model and can be assumed to be more representable for the results of the paper with regards to any external testing. 
These results are also confirmed in the comparisons of the disparity outputs. The pre-trained model and Monkaa model really underperform in comparison with the self-trained Scene Flow model. The Monkaa model mainly misses key details, whereas the pre-trained model lacks correct depth in the pictures.
The results on the external dataset make clear that the model does not provide us with satisfactory results with regards to this data. Thus given this dataset we conclude that the model in its current state is not a viable source for creating depth imagery in this scenario.

# Discussion
From our research it is made clear that the model provided by the authors of the paper does not yield desired results in the external dataset scenario. 
There are, however, a couple of things to consider if one wishes to use a model on this data.
Firstly, the model is trained on computer generated imagery and imagery from a car driving through streets. This was a logical choice as the authors wished to compete on a dataset of this nature in terms of disparity. However, if we are to use this data to determine disparity for motor blades in an unknown enclosed space, one can conclude that the model not being familiar with this scenario underperforms. One should be able to solve this issue by generating computer imagery of more motor blade and fan like 3D objects and potentially more data of actual motor blades with their corresponding disparity imagery. 
However, obtaining such a dataset would be quite costly. A simpler first step would be to finetune the Scene Flow model on the InStereo2K dataset [7]. We suspect that the model will learn better how to predict disparity for indoor natural stereo images, and this might have a beneficial effect for the purpose of motor blades. Sadly, we were not able to put this idea into practice, as our budget for Google Cloud was already depleted after training the models.

Secondly, our results are inconclusive on the fact of photos made through a prism are similar enough in nature to those made by a stereo camera. We do not have knowledge of the distance between the two pictures through the prism and how they are exactly related to each other. This is due to the fact that light through a prism can diverge and translate in many different ways. As such one should first test if it is even possible to use a model trained on stereo images can achieve similar results to a model with pictures through a prism.
Finally, one could consider further editing the external dataset such that the model can determine disparity and depth more easily. We think of methods such as increasing the contrast or removing the timestamp or removing the haze off of the imagery. 



# References
[1]  Chang,  J.  R.,  &  Chen,  Y.  S.  (2018).  Pyramid  stereo  matching  network. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 5410-5418).

[2]  Kendall,  A.,  Martirosyan,  H., Dasgupta,  S.,  Henry,  P., Kennedy,  R., Bachrach, A., & Bry, A. (2017). End-to-end learning of geometry and con-14
text for deep stereo regression. In Proceedings of the IEEE Internationa lConference on Computer Vision (pp. 66-75).

[3]  Mayer, N., Ilg, E., Hausser, P., Fischer, P., Cremers, D., Dosovitskiy, A., &  Brox,  T.  (2016).  A  large  dataset  to  train  convolutional  networks  for disparity,  optical  flow,  and  scene  flow  estimation.  In  Proceedings  of  theIEEE  conference  on  computer  vision  and  pattern  recognition  (pp.  4040-4048).

[4]  Gkolemis, V. (2018). Retrieved on 10-4-2020:  https://github.com /JiaRen-Chang/PSMNet/issues/64

[5]  Jang,  H.  (2020).  Retrieved  on  10-4-2020:   https://github.com  /JiaRen-Chang/PSMNet/issues/169

[6]  Pang, J., Sun, W., Ren, J. S., Yang, C., & Yan, Q. (2017). Cascade residual learning:  A two-stage convolutional neural network for stereo matching. In Proceedings  of  the  IEEE  International  Conference  on  Computer  Vision Workshops (pp. 887-895).

[7]  Bao, W., Wang, W., Xu, Y., Guo, Y., Hong, S., & Zhang, X. (2020). In-Stereo2K: a large real dataset for stereo matching in indoor scenes. Science China Information Sciences, 63(11), 1-11.

