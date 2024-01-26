This is the official implementation code for "DCP: Deep Channel Prior for Visual Recognition in Real-world Degradations"

# Deep Channel Prior
Deep Channel Prior (DCP) illustrates that the channel correlation matrix of features is an explicit means to reflect the corruption type of degraded images, while the feature itself can not represent its degradation type. DCP provides an explicit optimization direction for the unsupervised solution space by reducing the difference between the channel correlation matrices of degraded features and clear features.
![Deep Channel Prior from unpaired real clear and degraded images](https://github.com/liyuhang166/Deep_Channel_Prior/Fig2-Gram.png)

# Prepare data and weights path
You can download the training data (100 unpaired images) and test data (50,000 foggy images) of ImageNet-C (fog5) from the following path, as well as the corresponding weights：



# training
python train_VGG16.py 

python train_Resnet50.py

# testing 
python test_VGG16.py

python test_Resnet50.py


