import torch.nn.functional as F
import torch.nn as nn
import torch
class cnn2_zjc(nn.Module):#batchsize,16,16,16
    def __init__(self) :
        super(cnn2_zjc,self).__init__()

        self.cnn2_part0= nn.Sequential(nn.Conv2d(in_channels=16,out_channels=8,kernel_size=3,stride=1,padding=1),
                                     nn.BatchNorm2d(8),
                                     nn.ReLU()
                                     ) 
        self.cnn2_part1= nn.Sequential(nn.Conv2d(in_channels=8,out_channels=3,kernel_size=3,stride=1,padding=1),
                                nn.BatchNorm2d(3),
                                nn.ReLU()
                                ) 
        self.cnn2_part2= nn.Sequential(nn.Conv2d(in_channels=3,out_channels=3,kernel_size=3,stride=1,padding=1),
                        nn.BatchNorm2d(3),
                        nn.ReLU()
                        ) 
        self.cnn2_part3= nn.Sequential(nn.Conv2d(in_channels=3,out_channels=3,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(3)
                ) 
    def forward(self,x):
        x=self.cnn2_part0(x)
        x=F.interpolate(input=x,size=(32,32),mode='bilinear')
        x=self.cnn2_part1(x)
        x=F.interpolate(input=x,size=(64,64),mode='bilinear')
        x=self.cnn2_part2(x)
        x=F.interpolate(input=x,size=(128,128),mode='bilinear')
        x=self.cnn2_part3(x)
        return torch.sigmoid(x)
        