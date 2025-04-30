"""
Created on Mon Jun 17 2022
@author: fr
这份代码是将：

1. 在原MFB的卷积后加入RSTB
2. 在原上下两侧的U型网络 加入skip connection
3. 将原MFB的输入连接处更改，改为U型网络的后半段输入
4. 将原上下两侧的U型网络 的两层卷积 替换成 一层卷积+RSTB

"""


import torch
import torch.nn as nn
from model.STRB_fr import STRB_FR
import torch.nn.functional as F


def up(x):
    return F.interpolate(x, scale_factor=2)

def maxpool(in_channel,out_channel):
    # pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    pool = nn.Conv2d(in_channel,out_channel,2,2,padding=0)
    return pool


class MixedFusion_Block(nn.Module):  # MFB模块 ------------------

    def __init__(self, in_dim, out_dim, act_fn, input_resolution):
        super(MixedFusion_Block, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(in_dim * 3, in_dim, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(in_dim), act_fn, )

        # revised in 09/09/2019.
        # self.layer1 = nn.Sequential(nn.Conv2d(in_dim*3, in_dim,  kernel_size=1),nn.BatchNorm2d(in_dim),act_fn,)
        self.layer2 = nn.Sequential(nn.Conv2d(in_dim * 2, out_dim, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(out_dim), act_fn, )
        self.rstb_layer1 = STRB_FR(out_dim, input_resolution=input_resolution)

    def forward(self, x1, x2, xx):
        # multi-style fusion
        fusion_sum = torch.add(x1, x2)  # sum
        fusion_mul = torch.mul(x1, x2)  # 对两个张量进行逐元素乘法

        modal_in1 = torch.reshape(x1, [x1.shape[0], 1, x1.shape[1], x1.shape[2], x1.shape[3]])
        modal_in2 = torch.reshape(x2, [x2.shape[0], 1, x2.shape[1], x2.shape[2], x2.shape[3]])
        modal_cat = torch.cat((modal_in1, modal_in2), dim=1)
        fusion_max = modal_cat.max(dim=1)[0]

        out_fusion = torch.cat((fusion_sum, fusion_mul, fusion_max), dim=1)

        out1 = self.layer1(out_fusion)
        out2 = self.layer2(torch.cat((out1, xx), dim=1))

        out3 = self.rstb_layer1(out2)

        return out3


class MixedFusion_Block0(nn.Module):  # 第一个MFB模块，没有前面的输入，所以只有 x1 + x2
    def __init__(self, in_dim, out_dim, act_fn, input_resolution):
        super(MixedFusion_Block0, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(in_dim * 3, in_dim, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(in_dim), act_fn, )
        # self.layer1 = nn.Sequential(nn.Conv2d(in_dim*3, in_dim, kernel_size=1),nn.BatchNorm2d(in_dim),act_fn,)
        self.layer2 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(out_dim), act_fn, )
        self.rstb_layer1 = STRB_FR(out_dim,input_resolution=input_resolution)

    def forward(self, x1, x2):  # down_1_0m:[1,32,112,112] down_1_1m:[1,32,112,112]
        # multi-style fusion
        fusion_sum = torch.add(x1, x2)  # sum
        fusion_mul = torch.mul(x1, x2)

        modal_in1 = torch.reshape(x1, [x1.shape[0], 1, x1.shape[1], x1.shape[2], x1.shape[3]])
        modal_in2 = torch.reshape(x2, [x2.shape[0], 1, x2.shape[1], x2.shape[2], x2.shape[3]])
        modal_cat = torch.cat((modal_in1, modal_in2), dim=1)
        fusion_max = modal_cat.max(dim=1)[0]

        out_fusion = torch.cat((fusion_sum, fusion_mul, fusion_max), dim=1)  # [1,32*3,112,112]

        out1 = self.layer1(out_fusion)  # [1,32*3,112,112]----#[1,32,112,112]
        out2 = self.layer2(out1)  # [1,64,112,112]
        out3 = self.rstb_layer1(out2)

        return out3


##############################################
# define our model
class Multi_modal_generator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf):  # (1,1,32)
        super(Multi_modal_generator, self).__init__()

        self.in_dim = input_nc  # 1
        self.out_dim = ngf  # 32
        self.final_out_dim = output_nc  # 1

        act_fn = nn.LeakyReLU(0.2, inplace=True)
        act_fn2 = nn.ReLU(inplace=True)  # nn.ReLU()

        # ~~~ Encoding Paths ~~~~~~ #
        #######################################################################
        # Encoder **Modality 1
        #######################################################################

        self.down_1_0 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_dim, out_channels=self.out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim), act_fn,
            STRB_FR(dim=self.out_dim,input_resolution=(224,224))
        )
        self.pool_1_0 = maxpool(32,32)

        self.down_2_0 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim * 2), act_fn,
            STRB_FR(dim=self.out_dim * 2, input_resolution=(112, 112))
        )
        self.pool_2_0 = maxpool(64,64)

        self.down_3_0 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_dim * 2, out_channels=self.out_dim * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim * 4), act_fn,
            STRB_FR(dim=self.out_dim * 4, input_resolution=(56, 56))
        )
        self.pool_3_0 = maxpool(128,128)

        #######################################################################
        # Encoder **Modality 2
        #######################################################################
        self.down_1_1 = self.down_1_0
        self.pool_1_1 = maxpool(32,32)

        self.down_2_1 = self.down_2_0
        self.pool_2_1 = maxpool(64,64)

        self.down_3_1 = self.down_3_0
        self.pool_3_1 = maxpool(128,128)

        #######################################################################
        # ~~~ Encoding Paths ~~~~~~ #
        #######################################################################

        # ~~~ Decoding Path ~~~~~~ #
        # Modality 1
        self.up_1_0 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_dim * 4, out_channels=self.out_dim * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim * 4), act_fn2,
            STRB_FR(dim=self.out_dim * 4, input_resolution=(28, 28))
        )
        self.up_2_0 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_dim * 8, out_channels=self.out_dim * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim * 4), act_fn2,
            STRB_FR(dim=self.out_dim * 4, input_resolution=(56, 56))
        )
        self.up_3_0 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_dim * 4, out_channels=self.out_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim * 2), act_fn2,
            STRB_FR(dim=self.out_dim * 2, input_resolution=(56, 56))
        )
        self.up_4_0 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_dim * 4, out_channels=self.out_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim * 2), act_fn2,
            STRB_FR(dim=self.out_dim * 2, input_resolution=(112, 112))
        )
        self.up_5_0 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_dim * 2, out_channels=self.out_dim * 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim * 1), act_fn2,
            STRB_FR(dim=self.out_dim * 1, input_resolution=(112, 112))
        )
        self.up_6_0 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_dim * 2, out_channels=int(self.out_dim), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(self.out_dim)), act_fn2,
            STRB_FR(dim=int(self.out_dim), input_resolution=(224, 224))
        )
        self.out1 = nn.Sequential(nn.Conv2d(int(self.out_dim), 1, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU())

        # modality 2
        self.up_1_1 = self.up_1_0
        self.up_2_1 = self.up_2_0
        self.up_3_1 = self.up_3_0
        self.up_4_1 = self.up_4_0
        self.up_5_1 = self.up_5_0
        self.up_6_1 = self.up_6_0
        self.out2 = self.out1

        #######################################################################
        # fusion layer
        #######################################################################
        # down 1st layer
        self.down_fu_1 = MixedFusion_Block0(self.out_dim, self.out_dim * 2, act_fn,(112,112))
        self.pool_fu_1 = maxpool(64,64)

        self.down_fu_2 = MixedFusion_Block(self.out_dim * 2, self.out_dim * 4, act_fn,(56,56))
        self.pool_fu_2 = maxpool(128,128)

        self.down_fu_3 = MixedFusion_Block(self.out_dim * 4, self.out_dim * 4, act_fn, (28,28))
        # self.pool_fu_3 = maxpool()

        # down 4th layer
        self.down_fu_4 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_dim * 4, out_channels=self.out_dim * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim * 8), act_fn,
            STRB_FR(dim=self.out_dim * 8,input_resolution=(28,28))
        )

        self.fusion_up_1 = nn.Sequential(
                                         nn.Conv2d(self.out_dim * 8, self.out_dim * 4, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(self.out_dim * 4),
                                         act_fn2,
                                         STRB_FR(dim=self.out_dim * 4,input_resolution=(28,28))
        )
        self.fusion_up_2 = MixedFusion_Block(self.out_dim * 4, self.out_dim * 2, act_fn2, (28,28))
        self.fusion_up_3 = MixedFusion_Block(self.out_dim * 2, self.out_dim * 1, act_fn2, (56,56))
        self.fusion_up_4 = MixedFusion_Block(self.out_dim * 1, self.out_dim, act_fn2, (112,112))
        self.fusion_up_5 = nn.Sequential(
                                         nn.Conv2d(self.out_dim * 1, self.out_dim, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(self.out_dim),
                                         act_fn2,
                                         STRB_FR(dim=self.out_dim,input_resolution=(224,224))
        )
        self.out = nn.Sequential(nn.Conv2d(int(self.out_dim), 1, kernel_size=3, stride=1, padding=1),
                                 nn.ReLU())  # self.final_out_dim

    def forward(self, inputs):
        # ############################# #
        # i0 是将"x_fu = torch.cat([x1,x2],dim=1)" 里的 x1取出
        # i1 取出x2
        i0 = inputs[:, 0:1, :, :]
        i1 = inputs[:, 1:2, :, :]

        # -----  First Level --------
        down_1_0 = self.down_1_0(i0)  # [1,32,224,224]                                  +skip
        down_1_1 = self.down_1_1(i1)  # [1,32,224,224]                                  +skip

        # -----  Second Level --------
        # input_2nd = torch.cat((down_1_0,down_1_1,down_1_2,down_1_3),dim=1)
        # Max-pool
        down_1_0m = self.pool_1_0(down_1_0)  # [1,32,112,112]
        down_1_1m = self.pool_1_1(down_1_1)  # [1,32,112,112]

        down_2_0 = self.down_2_0(down_1_0m)  # [1,64,112,112]                                  +skip
        down_2_1 = self.down_2_1(down_1_1m)  # [1,64,112,112]                                  +skip


        # -----  Third Level --------
        # Max-pool
        down_2_0m = self.pool_2_0(down_2_0)  # [1,64,56,56]
        down_2_1m = self.pool_2_1(down_2_1)  # [1,64,56,56]

        down_3_0 = self.down_3_0(down_2_0m)  # [1,128,56,56]                                  +skip
        down_3_1 = self.down_3_1(down_2_1m)  # [1,128,56,56]                                  +skip

        # Max-pool
        down_3_0m = self.pool_3_0(down_3_0)  # [1,128,28,28]
        down_3_1m = self.pool_3_1(down_3_1)  # [1,128,28,28]

        # modality 1
        deconv_1_1 = self.up_1_0((down_3_0m))  # [1,128,28,28]
        up1_1 = up(deconv_1_1)  # [1,128,56,56]                                          #                                  +skip
        cat1_1 = torch.cat((down_3_0, up1_1), dim=1)  # [1,128*2,56,56]

        deconv_2_1 = self.up_2_0(cat1_1)  # [1,128,56,56]
        deconv_3_1 = self.up_3_0((deconv_2_1))  # [1,64,56,56]


        up2_1 = up(deconv_3_1)  # [1,64,112,112]                                       #                                  +skip
        cat2_1 = torch.cat((down_2_0, up2_1), dim=1)  # [1,64*2,112,112]

        deconv_4_1 = self.up_4_0(cat2_1)  # [1,64,112,112]
        deconv_5_1 = self.up_5_0((deconv_4_1))  # [1,32,112,112]


        up3_1 = up(deconv_5_1)  # [1,32,224,224]                                           #                                  +skip
        cat3_1 = torch.cat((down_1_0, up3_1), dim=1)  # [1,32*2,224,224]

        deconv_6_1 = self.up_6_0(cat3_1)  # [1,32,224,224]

        output1 = self.out1(deconv_6_1)  # [1,1,224,224]

        # modality 2
        deconv_1_2 = self.up_1_1((down_3_1m))  # [1,128,28,28]
        up1_2 = up(deconv_1_2)  # [1,128,56,56]                                          #                                  +skip
        cat1_2 = torch.cat((down_3_1, up1_2), dim=1)  # [1,128*2,56,56]

        deconv_2_2 = self.up_2_1(cat1_2)  # [1,128,56,56]
        deconv_3_2 = self.up_3_1((deconv_2_2))  # [1,64,56,56]

        up2_2 = up(deconv_3_2)  # [1,64,112,112]                                          #                                  +skip
        cat2_2 = torch.cat((down_2_1, up2_2), dim=1)  # [1,64*2,112,112]

        deconv_4_2 = self.up_4_1(cat2_2)  # [1,64,112,112]
        deconv_5_2 = self.up_5_1((deconv_4_2))  # [1,32,112,112]

        up3_2 = up(deconv_5_2)  # [1,32,224,224]                                          #                                  +skip
        cat3_2 = torch.cat((down_1_1, up3_2), dim=1)  # [1,32*2,224,224]

        deconv_6_2 = self.up_6_1(cat3_2)  # [1,32,224,224]

        output2 = self.out2(deconv_6_2)  # [1,1,224,224]

        # ----------------------------------------
        # fusion layer
        # ~~~~~~ Encoder
        down_fu_1 = self.down_fu_1(down_1_0m, down_1_1m)  # [1,64,112,112]
        down_fu_1m = self.pool_fu_1(down_fu_1)  # [1,64,56,56]

        down_fu_2 = self.down_fu_2(down_2_0m, down_2_1m, down_fu_1m)  # [1,128,56,56]
        down_fu_2m = self.pool_fu_2(down_fu_2)  # [1,128,28,28]

        down_fu_3 = self.down_fu_3(down_3_0m, down_3_1m, down_fu_2m)  # [1,128,28,28]

        down_fu_4 = self.down_fu_4(down_fu_3)  # [1,256,28,28]

        # ~~~~~~ Decoding
        deconv_1_0 = self.fusion_up_1(down_fu_4)  # [1,128,28,28]

        deconv_2_0 = self.fusion_up_2(deconv_1_1, deconv_1_2, deconv_1_0)  # [1,64,28,28]
        deconv_3_0 = self.fusion_up_3(deconv_3_1, deconv_3_2, up(deconv_2_0))  # [1,32,56,56]
        deconv_4_0 = self.fusion_up_4(deconv_5_1, deconv_5_2, up(deconv_3_0))  # [1,32,112,112]
        deconv_5_0 = self.fusion_up_5(up(deconv_4_0))  # [1,32,224,224]

        output = self.out(deconv_5_0)  # [1,1,224,224]

        return output, output1, output2


class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discrimintor_block(in_features, out_features, normalize=True):
            """Discriminator block"""
            layers = [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discrimintor_block(in_channels, 32, normalize=False),
            *discrimintor_block(32, 64),
            *discrimintor_block(64, 128),
            *discrimintor_block(128, 256),
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, kernel_size=3)
        )

    def forward(self, img):
        return self.model(img)

'''
检测模型结构
'''
# x= torch.rand([1,2,224,224])
#
# model = Multi_modal_generator(1, 1, 32)
#
# output, output1, output2 = model(x)
#
# print(output.shape,output1.shape,output2.shape)