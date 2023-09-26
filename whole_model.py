import torch.nn.functional as F
import torch.nn as nn
import torch
import time
from resnet34 import Resnet34
from cnn2 import cnn2_zjc
class wholemodel(nn.Module):
    def __init__(self,cnn1=Resnet34,cnn2=cnn2_zjc):
        super(wholemodel,self).__init__()
        self.cnn1=cnn1()
        self.cnn2=cnn2(inchanel=16,outchanel=3,stride=8)
        self.maxpool=nn.MaxPool2d(2)
        self.fc1=nn.Linear(524291,4096)
    def forward(self,x,deltac):
        output=self.cnn1(x) # 512 64 64
        output=self.maxpool(output)#512 32 32
        output=output.view(x.size(0),-1)#flatten  #32000000,
        #print("after view")
        #print("has already view")
        #concate the out1_img_fea_flat
        
        output=torch.cat((output,deltac),axis=1) #524291,
        output=self.fc1(output)
        #print(output)
        output=output.view(output.size(0),16,16,16)#batchsize,16,16,16
        output=self.cnn2(output)
        #print("output shape in whole model",output.shape)
        return output#3*1000*1000
    
    # def save(self, name=None):

    #    if name is None:
    #        prefix = 'checkpoints/' + '_'
    #        name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
    #    torch.save(self.state_dict(), name)
    #    return name
