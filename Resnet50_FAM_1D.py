import random
import torch
import torch.nn as nn
import os
import functools
from PIL import Image
from torch.nn import functional
from torch.optim import lr_scheduler
from torch.nn import init
from torch.utils.data import Dataset
from torchvision.models import resnet50
import itertools  
from block import Conv2dBlock  
from torch.nn import functional as F
from torch import Tensor
from typing import List
active_function = nn.PReLU()

# construct a dataset and return unpaired images
class MyDataset_train(Dataset):
    def __init__(self, input_root, label_root, transform_high, transform_low):
        # path
        self.input_root = input_root
        self.input_files = os.listdir(input_root)  # image files

        self.label_root = label_root
        self.label_files = os.listdir(label_root)

        self.transform_high = transform_high
        self.transform_low = transform_low

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        # random id
        a = random.randint(0, len(self.input_files)-1)
        b = random.randint(0, len(self.input_files)-1)

        # read degraded image
        input_img_path = os.path.join(self.input_root, self.input_files[a])
        input_img = Image.open(input_img_path).convert("RGB")

        # read clean image
        label_img_path = os.path.join(self.label_root, self.label_files[b])
        label_img = Image.open(label_img_path).convert("RGB")

        # data transform
        input_img = self.transform_low(input_img)
        label_img = self.transform_high(label_img)

        # returned unpaired images
        return (input_img, label_img)

# our Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = Conv2dBlock(256, 128, kernel_size=3, stride=1, padding=1, norm='bn', activation='relu', bias=False)
        self.conv2 = Conv2dBlock(128, 256, kernel_size=3, stride=1, padding=1, norm='bn', activation='relu', bias=False)
        self.conv3 = Conv2dBlock(256, 128, kernel_size=3, stride=1, padding=1, norm='bn', activation='relu', bias=False)
        
        self.conv4 = Conv2dBlock(256, 128, kernel_size=3, stride=1, padding=1, norm='bn', activation='relu', bias=False)
        self.conv5 = Conv2dBlock(128, 256, kernel_size=3, stride=1, padding=1, norm='bn', activation='relu', bias=False)
        self.conv6 = Conv2dBlock(256, 128, kernel_size=3, stride=1, padding=1, norm='bn', activation='relu', bias=False)

        self.conv1x1 = nn.Conv2d(128 + 256 + 128, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2x2 = nn.Conv2d(256 + 256 + 256, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU()
    
    def forward(self, feat):
        x = self.conv1(feat)
        feat1 = x
        x = self.conv2(x)
        feat2 = x
        x = self.conv3(x)
        x = torch.cat((x, feat1, feat2), 1)
        x = self.relu(self.conv1x1(x))
        feat3 = x

        x = self.conv4(x)
        feat4 = x
        x = self.conv5(x)
        feat5 = x
        x = self.conv6(x)
        x = torch.cat((x, feat4, feat5), 1)
        x = self.relu(self.conv1x1(x))

        x = torch.cat((x, feat3, feat), 1)
        x = self.relu(self.conv2x2(x))
        
        return x

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(512)
        self.leakyrelu1 = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d(512, 1024, kernel_size=4, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(1024)
        self.leakyrelu2 = nn.LeakyReLU(0.2, True)
        self.conv3 = nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_feature):
        conv1 = self.conv1(input_feature)
        bn1 = self.norm1(conv1)
        relu1 = self.leakyrelu1(bn1)
        conv2 = self.conv2(relu1)
        bn2 = self.norm2(conv2)
        relu2 = self.leakyrelu2(bn2)
        conv3 = self.sigmoid(self.conv3(relu2))     
        return conv3

# adversarial loss
class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))  
        self.register_buffer('fake_label', torch.tensor(target_fake_label))  
        self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)  

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class ResNet50_FAM(nn.Module):
    def __init__(self):
        super(ResNet50_FAM, self).__init__()
        # load pretrained model
        res50 = resnet50(pretrained=True)
        for pa in res50.parameters():
            pa.requires_grad = False
        self.conv1 = res50.conv1
        self.bn1 = res50.bn1
        self.relu = res50.relu
        self.maxpool = res50.maxpool
        self.layer1 = res50.layer1
        self.layer2 = res50.layer2
        self.layer3 = res50.layer3
        self.layer4 = res50.layer4
        self.avgpool = res50.avgpool
        self.fc = res50.fc
        # Hyperparameters
        self.lambda_adv = 5.0
        self.lambda_cyc = 10.0
        self.lambda_idt = 5.0
        self.MultiGPU = True
        self.lr_G = 0.0002
        self.lr_D = 0.0001
        # GD2C GC2D DC DD
        self.GD2C = Generator()
        self.GC2D = Generator()
        self.DC = Discriminator()  # Discriminator
        self.DD = Discriminator()  
        # init modules' weights
        self.init_net(self.GD2C, init_type='kaiming', init_gain=0.02, gpu_ids=self.MultiGPU)
        self.init_net(self.GC2D, init_type='kaiming', init_gain=0.02, gpu_ids=self.MultiGPU)
        self.init_net(self.DC, init_type='kaiming', init_gain=0.02, gpu_ids=self.MultiGPU)
        self.init_net(self.DD, init_type='kaiming', init_gain=0.02, gpu_ids=self.MultiGPU)
        # define three losses in stage-1
        self.Adv_loss = GANLoss().cuda()
        self.Cyc_loss = torch.nn.L1Loss()
        self.Idt_loss = torch.nn.L1Loss()
        # optimizer
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.GD2C.parameters(), self.GC2D.parameters()),
                                            lr=self.lr_G, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.DC.parameters(), self.DD.parameters()),
                                            lr=self.lr_D, betas=(0.5, 0.999))                       
        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
        # learning rate schedule
        self.schedulers = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) for optimizer in self.optimizers]

    # SPL
    def Resnet50_forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)

        return x

    def forward(self, haze, clean):

        DF = self.Resnet50_forward(haze)
        CF = self.Resnet50_forward(clean)

        fake_DF_1 = self.GD2C(DF)  # fake_CF
        fake_DF_2 = self.GC2D(fake_DF_1)  # fake_fake_DF

        fake_CF_1 = self.GC2D(CF)  # fake_DF
        fake_CF_2 = self.GD2C(fake_CF_1) # fake_fake_CF

        return DF, CF, fake_DF_1, fake_DF_2, fake_CF_1, fake_CF_2

    # optimize generators and discriminators
    def optimize(self, clean, haze, DF, CF, fake_DF_1, fake_DF_2, fake_CF_1, fake_CF_2):
        # GD2C、GC2D
        self.set_requires_grad([self.DC, self.DD], False)
        self.optimizer_G.zero_grad()
        loss_G = self.backward_G(DF, CF, fake_DF_1, fake_DF_2, fake_CF_1, fake_CF_2)
        self.optimizer_G.step()
        
        # DC、DD
        self.set_requires_grad([self.DC, self.DD], True)
        self.optimizer_D.zero_grad()
        loss_DC = self.backward_DC(CF, fake_DF_1) 
        loss_DD = self.backward_DD(DF, fake_CF_1)
        self.optimizer_D.step()
       
        return loss_G, loss_DC, loss_DD

    # update generators' parameters
    def backward_G(self, DF, CF, fake_DF_1, fake_DF_2, fake_CF_1, fake_CF_2):
        # L_adv
        Adv_loss_GD2C = self.Adv_loss(self.DC(fake_DF_1), True) * self.lambda_adv
        Adv_loss_GC2D = self.Adv_loss(self.DD(fake_CF_1), True) * self.lambda_adv
        # L_cyc
        Cycle_loss_DF = self.Cyc_loss(DF, fake_DF_2) * self.lambda_cyc
        Cycle_loss_CF = self.Cyc_loss(CF, fake_CF_2) * self.lambda_cyc
        # L_idt
        fake_DF_3 = self.GC2D(DF)
        Idt_loss_GC2D = self.Idt_loss(DF, fake_DF_3) * self.lambda_idt
        fake_CF_3 = self.GD2C(CF)
        Idt_loss_GD2C = self.Idt_loss(CF, fake_CF_3) * self.lambda_idt
        # combined loss
        loss_G = Adv_loss_GD2C + Adv_loss_GC2D + Cycle_loss_DF + Cycle_loss_CF + Idt_loss_GC2D + Idt_loss_GD2C
        loss_G.backward()
        return loss_G

    # update discriminator's parameters
    def backward_DC(self, CF, fake_DF_1):
        # Real
        loss_DC_real = self.Adv_loss(self.DC(CF), True)
        # Fake
        loss_DC_fake = self.Adv_loss(self.DC(fake_DF_1.detach()), False)
        # Combined loss
        loss_DC = (loss_DC_real + loss_DC_fake) * 0.5
        # backward
        loss_DC.backward()
        return loss_DC
    
    def backward_DD(self, DF, fake_CF_1):
        # Real
        loss_DD_real = self.Adv_loss(self.DD(DF), True)
        # Fake
        loss_DD_fake = self.Adv_loss(self.DD(fake_CF_1.detach()), False)
        # Combined loss
        loss_DD = (loss_DD_real + loss_DD_fake) * 0.5
        # backward
        loss_DD.backward()
        return loss_DD

    # init modules' weights
    def init_net(self, net, init_type='normal', init_gain=0.02, gpu_ids=False):
        net.cuda()
        if gpu_ids:
            net = torch.nn.DataParallel(net)  # 多GPU训练
        self.init_weights(net, init_type, gain=init_gain)
        return net

    def init_weights(self, net, init_type='normal', gain=0.02):  # gain是采样分布可选的缩放因子
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'zeros':
                    init.zeros_(m.weight.data)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    # original resnet50 forward path
    def test_forward(self, x):
        # SPL
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    # ResNet50_UFEM forward path
    def test_forward1(self, x):
        # SPL
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        
        # GD2C enhance
        self.GD2C.eval()
        x = self.GD2C(x)

        # DPL
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    # get unenhanced shallow features
    def get_feature(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)

        return x

    # get enhanced shallow features
    def get_enh_feature(self, x):
        # SPL
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)

        # GD2C enhance
        self.GD2C.eval()
        x = self.GD2C(x)

        return x

 