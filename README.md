Official implementation code for "DCP: Deep Channel Prior for Visual Recognition in Real-world Degradations"

# Deep Channel Prior
Deep Channel Prior (DCP) illustrates that the channel correlation matrix of features is an explicit means to reflect the corruption type of degraded images, while the feature itself can not represent its degradation type. DCP provides an explicit optimization direction for the unsupervised solution space by reducing the difference between the channel correlation matrices of degraded features and clear features.
![Deep Channel Prior from unpaired real clear and degraded images](https://github.com/liyuhang166/Deep_Channel_Prior/blob/main/Fig2-Gram2.png)

# Prepare data and weights
You can download the training data (100 unpaired images) and test data (50,000 foggy images) of ImageNet-C (fog5) from the following path, as well as the corresponding weights：
https://drive.google.com/drive/folders/1Q84HLFpjAFq91NG21piR-doxCTQHPU3q?usp=drive_link

# Structure of generators and discriminators
For the generator, we employed two different structures overall. Specifically, in terms of the experiments on real datasets, we mainly utilized a U-Net style generator structure. The encoder consists of the input convolutional layer, two downsampling layers, and the dilated convolutional layer, while the decoder comprises two upsampling layers and the output convolutional layer. In terms of the experiments on synthetic datasets, we mainly adopted a flattened generator structure with residual connections. It consists of two basic blocks and a 1×1 convolutional layer, with each basic block containing several convolutional layers. For the discriminator, we exclusively utilized a PatchGAN discriminator, which comprises three convolutional layers and a sigmoid output layer.
![Specific structure of the generators and discriminators](https://github.com/liyuhang166/Deep_Channel_Prior/blob/main/Generator.png)

# Training
python train_VGG16.py 

python train_Resnet50.py

# Testing 
python test_VGG16.py

python test_Resnet50.py


