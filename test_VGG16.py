from datetime import datetime
from torch.utils.data import DataLoader
from torchvision import datasets
import torch
import torch.nn as nn
import os
import numpy as np
import random
from torchvision import transforms
from torchvision.models import vgg16

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(10)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_testdataset():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

def TestDirect(images_path, ModelPath):

    valid_bs = 64
    validTransform = get_testdataset()
    test_dir = images_path
    test_datasets = datasets.ImageFolder(test_dir, transform=validTransform)
    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=valid_bs, shuffle=True)

    # load our model with UFEM
    NetEn = torch.load(ModelPath)
    NetEn = NetEn.cuda().eval()

    # loss
    loss_func = nn.CrossEntropyLoss()
    test_loss = 0.
    test_acc = 0.

    for batch_x, batch_y in test_dataloader:
        with torch.no_grad():
            # cuda 
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            # forward
            out = NetEn.test_forward1(batch_x)
            loss = loss_func(out, batch_y)
            test_loss += loss.item()
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum()
            test_acc += num_correct.item()

    print('Vgg16_FAM_Test_Loss: {:.6f}, Vgg16_FAM_Acc: {:.6f}'.format(test_loss / (len(
        test_datasets)), test_acc / (len(test_datasets))))


def test_vgg16(images_path):

    valid_bs = 128
    validTransform = get_testdataset()
    test_dir = images_path
    test_datasets = datasets.ImageFolder(test_dir, transform=validTransform)
    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=valid_bs, shuffle=True)

    # load model
    NetEn = vgg16(pretrained=False).cuda()
    state = torch.load('vgg16-397923af.pth')
    NetEn.load_state_dict(state)
    NetEn = NetEn.cuda().eval()

    # loss
    loss_func = nn.CrossEntropyLoss()
    test_loss = 0.
    test_acc = 0.

    for batch_x, batch_y in test_dataloader:
        with torch.no_grad():
            # cuda
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            # forward
            out = NetEn(batch_x)
            loss = loss_func(out, batch_y)
            test_loss += loss.item()
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum()
            test_acc += num_correct.item()

    print('Vgg16_Test_Loss: {:.6f}, Vgg16_Acc: {:.6f}'.format(test_loss / (len(
        test_datasets)), test_acc / (len(test_datasets))))

if __name__ == '__main__':

    ModelPath = ['weights/VGG16/fog/5/100/3D/Vgg_FAM_10epoch.pt',  'weights/VGG16/fog/5/100/3D/Vgg_FAM_20epoch.pt',  'weights/VGG16/fog/5/100/3D/Vgg_FAM_30epoch.pt',
                 'weights/VGG16/fog/5/100/3D/Vgg_FAM_40epoch.pt',  'weights/VGG16/fog/5/100/3D/Vgg_FAM_50epoch.pt',  'weights/VGG16/fog/5/100/3D/Vgg_FAM_60epoch.pt',
                 'weights/VGG16/fog/5/100/3D/Vgg_FAM_70epoch.pt',  'weights/VGG16/fog/5/100/3D/Vgg_FAM_80epoch.pt',  'weights/VGG16/fog/5/100/3D/Vgg_FAM_90epoch.pt',
                 'weights/VGG16/fog/5/100/3D/Vgg_FAM_100epoch.pt', 'weights/VGG16/fog/5/100/3D/Vgg_FAM_110epoch.pt', 'weights/VGG16/fog/5/100/3D/Vgg_FAM_120epoch.pt',
                 'weights/VGG16/fog/5/100/3D/Vgg_FAM_130epoch.pt', 'weights/VGG16/fog/5/100/3D/Vgg_FAM_140epoch.pt', 'weights/VGG16/fog/5/100/3D/Vgg_FAM_150epoch.pt',
                 'weights/VGG16/fog/5/100/3D/Vgg_FAM_160epoch.pt', 'weights/VGG16/fog/5/100/3D/Vgg_FAM_170epoch.pt', 'weights/VGG16/fog/5/100/3D/Vgg_FAM_180epoch.pt',
                 'weights/VGG16/fog/5/100/3D/Vgg_FAM_190epoch.pt', 'weights/VGG16/fog/5/100/3D/Vgg_FAM_200epoch.pt']

    images_path = 'data/test_fam/ImageNet/haze_img_val/fog/5' 
    print("ImageNet-C and ImageNet Test on Vgg16 and Vgg16_FAM")

    # test original vgg16
    print("原始Vgg16的检测精度为: ")
    test_vgg16(images_path) 

    # test our model
    for i in range(1, 11):
        print("Vgg16_" + str(i * 10) + "epochs的检测精度为: ") 
        TestDirect(images_path, ModelPath[i-1])


