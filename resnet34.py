
import torch.nn as nn
from torch.nn import functional as F
class ResidualBlock(nn.Module): #3*1000*1000
    def __init__(self,right_conv,input_chanel,output_chanel,stride):
        super(ResidualBlock,self).__init__()
        self.right_conv=right_conv
        self.input_chanel=input_chanel
        self.output_chanel=output_chanel
        self.stride=stride
        
        self.residual=nn.Sequential(
            nn.Conv2d(self.input_chanel,self.output_chanel,3,1,1),
            nn.BatchNorm2d(self.output_chanel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.output_chanel,output_chanel,3,1,1),
            nn.BatchNorm2d(self.output_chanel)            
        )
        if self.right_conv==True:
            self.right=nn.Sequential(nn.Conv2d(self.input_chanel,self.output_chanel,kernel_size=1,stride=1),nn.BatchNorm2d(self.output_chanel))
        else:  
            self.right=None

    def forward(self,x):
       
        out1=self.residual(x)
        # print("x shape",x.shape)
        #print("out1 shape",out1.shape)
        if self.right_conv==False:
            finalout=out1+x
        else:
            finalout=out1+self.right(x)
           # print("finalout shape",finalout.shape)
        return F.relu(finalout)


class Resnet34(nn.Module):
    
    def __init__(self):
        super(Resnet34,self).__init__()#3*128*128
        self.pre=nn.Sequential(nn.Conv2d(3,64,8,2,3),nn.BatchNorm2d(64))#64 64 64
        
        self.layer1 = self.makelayer( 64, 128, True,2,2)#128 64 64
        self.layer2 = self.makelayer( 128, 256, True,2, 2)#256 64 64
        self.layer3 = self.makelayer( 256, 512, True,2, 2)#415 64 64
        self.layer4 = self.makelayer( 512, 512, True,2, 2)#512 64 64
        
        
        
    def makelayer(self,input_chanel,output_chanel,right_conv,block_num,stide):#make layer 中第一层可以是升维的，后面的得是不变的
        layers=[]
        layers.append(ResidualBlock(True,input_chanel,output_chanel,1))
        for i in range(block_num-1):
            layers.append(ResidualBlock(right_conv,output_chanel,output_chanel,stide))
        
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x=self.pre(x)
       # print("pre",x.shape)
        x=self.layer1(x)
        #print("afterlayer1",x.shape)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        return x
        