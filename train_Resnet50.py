import torch.utils.data
from torch.utils.data import DataLoader
from torch.autograd import Variable
# You should import ResNet50_FAM from different .py files in different stages
# Resnet50_FAM_3D.py for stage-1, Resnet50_FAM_Domain.py for S1+S2, Resnet50_FAM_Domain_Direct.py for only S2
from Resnet50_FAM_3D import ResNet50_FAM, MyDataset_train
import torch
import os
import torchvision
from torch import Tensor
from typing import List
from torchvision import transforms
import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_traindataset():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

setup_seed(10)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def TrainFunction(train_path1, train_path2, Epoch):
    train_bs = 5
    max_epoch = Epoch

    # data transform
    high_Transform = get_traindataset()
    low_Transform = get_traindataset()

    # build dataset
    train_data = MyDataset_train(input_root=train_path1, label_root=train_path2, transform_high=high_Transform,
                                 transform_low=low_Transform)
    # dataLoder
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=train_bs, num_workers=32, shuffle=True)

    # init our model with UFEM
    NetEn = ResNet50_FAM().cuda()

    # training
    for epoch in range(max_epoch):
        NetEn.train()
        for i, data in enumerate(train_loader):
            # get unpaired images
            inputsLow, inputsHigh = data
            inputsHigh, inputsLow = Variable(inputsHigh), Variable(inputsLow)
            # cuda
            inputsHigh = inputsHigh.cuda()
            inputsLow = inputsLow.cuda()

            # ------------ S1 training -------------

            # stage-1, three Discriminators(3D)
            DF, CF, fake_DF_1, fake_DF_2, fake_CF_1, fake_CF_2, real_clean_dis1, real_haze_dis1, fake_clean_dis1, fake_haze_dis1, real_clean_dis2, real_haze_dis2, fake_clean_dis2, fake_haze_dis2 = NetEn(inputsLow, inputsHigh)
            loss_G, loss_DC, loss_DD = NetEn.optimize(inputsHigh, inputsLow, 
                                    DF, CF, fake_DF_1, fake_DF_2, fake_CF_1, fake_CF_2, real_clean_dis1, real_haze_dis1, fake_clean_dis1, fake_haze_dis1, real_clean_dis2, real_haze_dis2, fake_clean_dis2, fake_haze_dis2)
            
            # stage-1, two Discriminators(2D)
            # DF, CF, fake_DF_1, fake_DF_2, fake_CF_1, fake_CF_2, real_clean_dis, real_haze_dis, fake_clean_dis, fake_haze_dis = NetEn(inputsLow, inputsHigh)
            # loss_G, loss_DC, loss_DD = NetEn.optimize(inputsHigh, inputsLow, 
            #                                             DF, CF, fake_DF_1, fake_DF_2, fake_CF_1, fake_CF_2, real_clean_dis, real_haze_dis, fake_clean_dis, fake_haze_dis)

            # stage-1, one Discriminator(1D)
            # DF, CF, fake_DF_1, fake_DF_2, fake_CF_1, fake_CF_2 = NetEn(inputsLow, inputsHigh)
            # loss_G, loss_DC, loss_DD = NetEn.optimize(inputsHigh, inputsLow, DF, CF, fake_DF_1, fake_DF_2, fake_CF_1, fake_CF_2)

            # stage-1 print
            print(
                "Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Generator_loss: {:.4f} DC_loss: {:.4f} DD_loss: {:.4f}".format(
                    epoch + 1, max_epoch, i + 1, len(train_loader), loss_G.item(), loss_DC.item(), loss_DD.item()))

            # ------------ S1+S2 / S2 training -------------

            # S1+S2, when you train stage-2 immediately after stage-1, comment out the above code and uncomment this part of the code
            # DF, CF, fake_DF_1, fake_DF_2 = NetEn(inputsLow, inputsHigh)
            # loss_G, loss_D = NetEn.optimize(inputsHigh, inputsLow, DF, CF, fake_DF_1, fake_DF_2)

            # only stage-2
            # DF, CF, fake_DF_1 = NetEn(inputsLow, inputsHigh)
            # loss_G, loss_D = NetEn.optimize(inputsHigh, inputsLow, DF, CF, fake_DF_1)

            # S1+S2 print
            # print(
            #     "Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Generator_loss: {:.4f} Ddomain_loss: {:.4f}".format(
            #         epoch + 1, max_epoch, i + 1, len(train_loader), loss_G.item(), loss_D.item()))

        # save models
        if epoch % 10 == 0 and epoch != 0:
            torch.save(NetEn, 'weights/ResNet50/fog/5/100/3D/Resnet50_FAM_' + str(epoch) + 'epoch.pt')

if __name__ == '__main__':

    # two unpaired imagesets' paths, train_path1 for degraded images, train_path2 for clean images
    train_path1 = 'data/train_fam/train_fam_imgnet/images_haze/100/fog/5'
    train_path2 = 'data/train_fam/train_fam_imgnet/images_clean/100/version1/'
    epoch = 201
    print("ImageNet and ImageNet-C Train On Resnet50_FAM")

    TrainFunction(train_path1, train_path2, epoch)