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
from torchvision.models import vgg16
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


# our Generator1, GD2C
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = Conv2dBlock(64, 128, kernel_size=3, stride=1, padding=1, norm='bn', activation='relu', bias=False)
        self.conv2 = Conv2dBlock(128, 256, kernel_size=3, stride=1, padding=1, norm='bn', activation='relu', bias=False)
        self.conv3 = Conv2dBlock(256, 128, kernel_size=3, stride=1, padding=1, norm='bn', activation='relu', bias=False)
        
        self.conv4 = Conv2dBlock(64, 128, kernel_size=3, stride=1, padding=1, norm='bn', activation='relu', bias=False)
        self.conv5 = Conv2dBlock(128, 256, kernel_size=3, stride=1, padding=1, norm='bn', activation='relu', bias=False)
        self.conv6 = Conv2dBlock(256, 128, kernel_size=3, stride=1, padding=1, norm='bn', activation='relu', bias=False)

        self.conv1x1 = nn.Conv2d(128 + 256 + 128, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2x2 = nn.Conv2d(64 + 64 + 64, 64, kernel_size=1, stride=1, padding=0, bias=False)
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


# our Generator2, GE2C
class Generator2(nn.Module):
    def __init__(self):
        super(Generator2, self).__init__()
        self.conv1 = Conv2dBlock(64, 256, kernel_size=3, stride=1, padding=1, norm='bn', activation='relu', bias=False)
        self.conv2 = Conv2dBlock(256, 64, kernel_size=3, stride=1, padding=1, norm='bn', activation='relu', bias=False)

        self.conv1x1 = nn.Conv2d(64 + 256 + 64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU()

    def forward(self, feat):
        x = self.conv1(feat)
        feat1 = x
        x = self.conv2(x)

        x = torch.cat((x, feat1, feat), 1)
        x = self.relu(self.conv1x1(x))

        return x

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(64, 512, kernel_size=4, stride=1, padding=1, bias=False)
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


class Vgg16_FAM(nn.Module):
    def __init__(self):
        super(Vgg16_FAM, self).__init__()
        # load pretrained model
        vgg = vgg16(pretrained=False).cuda()
        state = torch.load('vgg16-397923af.pth')
        vgg.load_state_dict(state)
        for pa in vgg.parameters():
            pa.requires_grad = False
        # vgg16
        self.feature0 = vgg.features[0]
        self.feature1 = vgg.features[1]
        self.feature2 = vgg.features[2]
        self.feature3 = vgg.features[3]      # Conv1_2
        self.feature4 = vgg.features[4]      # maxpool
        self.feature5 = vgg.features[5]      
        self.feature6 = vgg.features[6]      
        self.feature7 = vgg.features[7]
        self.feature8 = vgg.features[8]      # Conv2_2
        self.feature9 = vgg.features[9]      # maxpool
        self.feature10 = vgg.features[10]    
        self.feature11 = vgg.features[11]    
        self.feature12 = vgg.features[12]
        self.feature13 = vgg.features[13]
        self.feature14 = vgg.features[14]
        self.feature15 = vgg.features[15]    # Conv3_3
        self.feature16 = vgg.features[16]    
        self.feature17 = vgg.features[17]
        self.feature18 = vgg.features[18]
        self.feature19 = vgg.features[19]
        self.feature20 = vgg.features[20]
        self.feature21 = vgg.features[21]
        self.feature22 = vgg.features[22]
        self.feature23 = vgg.features[23]
        self.feature24 = vgg.features[24]
        self.feature25 = vgg.features[25]
        self.feature26 = vgg.features[26]
        self.feature27 = vgg.features[27]
        self.feature28 = vgg.features[28]
        self.feature29 = vgg.features[29]
        self.feature30 = vgg.features[30]
        self.avgpool = vgg.avgpool
        self.classifier = vgg.classifier
        # Hyperparameters
        self.lambda_adv = 5.0
        self.lambda_con = 10.0
        self.lambda_sty = 1000.0
        self.MultiGPU = False
        self.lr_G = 0.0002
        self.lr_D = 0.0001
        # G and D
        self.GD2C = Generator()     # GD2C
        self.GDomain = Generator2() # GE2C
        self.Ddomain = Discriminator()
        # init modules' weights
        self.GD2C = Generator()
        model = torch.load('weights/VGG16/fog/5/100/3D/0.345440.pt')  # model produced in stage-1
        self.GD2C = model.GD2C  # load GD2C
        self.init_net(self.GDomain, init_type='kaiming', init_gain=0.02, gpu_ids=self.MultiGPU)
        self.init_net(self.Ddomain, init_type='kaiming', init_gain=0.02, gpu_ids=self.MultiGPU)   
        # define three losses in stage-2
        self.Adv_loss = GANLoss().cuda()
        self.Con_loss = torch.nn.L1Loss().cuda()
        self.Sty_loss = torch.nn.L1Loss().cuda()
        # optimizer
        self.optimizer_G = torch.optim.Adam(self.GDomain.parameters(), lr=self.lr_G, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.Ddomain.parameters(), lr=self.lr_D, betas=(0.5, 0.999))                     
        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
        # learning rate schedule
        self.schedulers = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) for optimizer in self.optimizers]

    # SPL
    def Vgg_forward(self, x):
        x = self.feature0(x)
        x = self.feature1(x)
        x = self.feature2(x)
        x = self.feature3(x)    # Conv1_2
        
        return x
    
    # channel correlation matrix
    def ChannelMatrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        return G.div(a * b * c * d)

    # correlation loss 
    def Vgg16_forward_for_style(self, x):
        x1 = x
        x = self.feature4(x1)
        x = self.feature5(x)
        x = self.feature6(x)
        x = self.feature7(x)
        x2 = self.feature8(x)    # Conv2_2
        x = self.feature9(x2)
        x = self.feature10(x)
        x = self.feature11(x)
        x = self.feature12(x)
        x = self.feature13(x)
        x = self.feature14(x)
        x3 = self.feature15(x)   # Conv3_3
        x = self.feature16(x3)
        x = self.feature17(x)
        x = self.feature18(x)
        x = self.feature19(x)
        x = self.feature20(x)
        x = self.feature21(x)
        x4 = self.feature22(x)   # Conv4_3

        x1 = self.ChannelMatrix(x1)
        x2 = self.ChannelMatrix(x2)
        x3 = self.ChannelMatrix(x3)
        x4 = self.ChannelMatrix(x4)

        return x1, x2, x3, x4

    # content
    def Vgg16_forward_for_content(self, x):
        x = self.feature4(x)
        x = self.feature5(x)
        x = self.feature6(x)
        x = self.feature7(x)
        x = self.feature8(x)    # Conv2_2
        x = self.feature9(x)
        x = self.feature10(x)
        x = self.feature11(x)
        x = self.feature12(x)
        x = self.feature13(x)
        x = self.feature14(x)
        x = self.feature15(x)   # Conv3_3
        x = self.feature16(x)
        x = self.feature17(x)
        x = self.feature18(x)
        x = self.feature19(x)
        x = self.feature20(x)
        x = self.feature21(x)
        x = self.feature22(x)   # Conv4_3

        return x

    def forward(self, haze, clean):
        DF = self.Vgg_forward(haze)
        CF = self.Vgg_forward(clean)

        fake_DF_1 = self.GD2C(DF) 
        fake_DF_2 = self.GDomain(fake_DF_1)

        return DF, CF, fake_DF_1, fake_DF_2

    # optimize generators and discriminators
    def optimize(self, clean, haze, DF, CF, fake_DF_1, fake_DF_2):
        # GE2C
        self.set_requires_grad([self.GD2C, self.Ddomain], False)
        self.optimizer_G.zero_grad()
        loss_G = self.backward_G(DF, CF, fake_DF_1, fake_DF_2)
        self.optimizer_G.step()
        
        # Discriminator
        self.set_requires_grad(self.Ddomain, True)
        self.optimizer_D.zero_grad()
        loss_D = self.backward_D(CF, fake_DF_2) 
        self.optimizer_D.step()
       
        return loss_G, loss_D

    # only optimize GE2C, freeze GD2C
    def backward_G(self, DF, CF, fake_DF_1, fake_DF_2):
        # L_adv
        Adv_loss = self.Adv_loss(self.Ddomain(fake_DF_2), True) * self.lambda_adv
        # L_correlation
        style_DF_1, style_DF_2, style_DF_3, style_DF_4 = self.Vgg16_forward_for_style(fake_DF_2)
        style_CF_1, style_CF_2, style_CF_3, style_CF_4 = self.Vgg16_forward_for_style(CF)
        Style_loss_1 = self.Sty_loss(style_DF_1.mul_(self.lambda_sty), style_CF_1.mul_(self.lambda_sty))
        Style_loss_2 = self.Sty_loss(style_DF_2.mul_(self.lambda_sty), style_CF_2.mul_(self.lambda_sty))
        Style_loss_3 = self.Sty_loss(style_DF_3.mul_(self.lambda_sty), style_CF_3.mul_(self.lambda_sty))
        Style_loss_4 = self.Sty_loss(style_DF_4.mul_(self.lambda_sty), style_CF_4.mul_(self.lambda_sty))
        Style_loss = 1 * Style_loss_1 + 2 * Style_loss_2 + 3 * Style_loss_3 + 4 * Style_loss_4
        # L_content
        content_DF_1 = self.Vgg16_forward_for_content(fake_DF_1)
        content_DF_2 = self.Vgg16_forward_for_content(fake_DF_2)
        Content_loss = self.Con_loss(content_DF_2, content_DF_1) * self.lambda_con
        # combined loss
        loss_G = Adv_loss + Style_loss + Content_loss
        loss_G.backward()
        return loss_G

    # optimize Discriminator
    def backward_D(self, CF, fake_DF_2):
        # Real
        loss_D_real = self.Adv_loss(self.Ddomain(CF), True)
        # Fake
        loss_D_fake = self.Adv_loss(self.Ddomain(fake_DF_2.detach()), False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    # init modules'weights
    def init_net(self, net, init_type='normal', init_gain=0.02, gpu_ids=False):
        net.cuda()
        if gpu_ids:
            net = torch.nn.DataParallel(net)  
        self.init_weights(net, init_type, gain=init_gain)
        return net

    def init_weights(self, net, init_type='normal', gain=0.02):  
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

    # original vgg16 forward path
    def test_forward(self, x):
        # SPL
        x = self.feature0(x)
        x = self.feature1(x)
        x = self.feature2(x)
        x = self.feature3(x)
        
        # DPL
        x = self.feature4(x)
        x = self.feature5(x)
        x = self.feature6(x)
        x = self.feature7(x)
        x = self.feature8(x)
        x = self.feature9(x)
        x = self.feature10(x)
        x = self.feature11(x)
        x = self.feature12(x)
        x = self.feature13(x)
        x = self.feature14(x)
        x = self.feature15(x)
        x = self.feature16(x)
        x = self.feature17(x)
        x = self.feature18(x)
        x = self.feature19(x)
        x = self.feature20(x)
        x = self.feature21(x)
        x = self.feature22(x)
        x = self.feature23(x)
        x = self.feature24(x)
        x = self.feature25(x)
        x = self.feature26(x)
        x = self.feature27(x)
        x = self.feature28(x)
        x = self.feature29(x)
        x = self.feature30(x)

        # classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    # VGG16_UFEM forward path
    def test_forward1(self, x):
        # SPL
        x = self.feature0(x)
        x = self.feature1(x)
        x = self.feature2(x)
        x = self.feature3(x)
        
        # UFEM (GD2C + GE2C) enhance
        self.GD2C.eval()
        self.GDomain.eval()
        x = self.GD2C(x)
        x = self.GDomain(x)
        
        # DPL
        x = self.feature4(x)
        x = self.feature5(x)
        x = self.feature6(x)
        x = self.feature7(x)
        x = self.feature8(x)
        x = self.feature9(x)
        x = self.feature10(x)
        x = self.feature11(x)
        x = self.feature12(x)
        x = self.feature13(x)
        x = self.feature14(x)
        x = self.feature15(x)
        x = self.feature16(x)
        x = self.feature17(x)
        x = self.feature18(x)
        x = self.feature19(x)
        x = self.feature20(x)
        x = self.feature21(x)
        x = self.feature22(x)
        x = self.feature23(x)
        x = self.feature24(x)
        x = self.feature25(x)
        x = self.feature26(x)
        x = self.feature27(x)
        x = self.feature28(x)
        x = self.feature29(x)
        x = self.feature30(x)

        # classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    # get unenhanced shallow features
    def get_feature(self, x):
        x = self.feature0(x)
        x = self.feature1(x)
        x = self.feature2(x)
        x = self.feature3(x)

        return x

    # get enhanced shallow features
    def get_enh_feature(self, x):
        # SPL
        x = self.feature0(x)
        x = self.feature1(x)
        x = self.feature2(x)
        x = self.feature3(x)
        
        # UFEM enhance
        self.GD2C.eval()
        self.GDomain.eval()
        x = self.GD2C(x)
        x = self.GDomain(x)

        return x

 