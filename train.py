# import sys
# sys.path.append('/mnt/d/junch_data/test_junch')
import torch
from data.data_loader import data_loader
# from yeah import wholemodel
from torch import nn
from whole_model import wholemodel



def train():
    
    train_dataloader=data_loader(data_dir='/mnt/d/junch_data/test_junch/model/data/train/',batch_size=4)
    #test_dataloader=data_loader(data_dir='/mnt/d/junch_data/test_junch/model/data/test',batch_size=4)

    device=torch.device("cuda:0")
    
    
    total_train_step = 0
    epoch = 10
    loss_fn=nn.MSELoss()
    model=wholemodel().to(device)
    optimizer=torch.optim.SGD(params=model.parameters(),lr=1e-3,momentum=0.9)
    
    save_model_every_epoch=1
    save_path='/mnt/d/junch_data/test_junch/model/save_model/checkpoint/'
    #epoch
    for i in range(epoch):
        torch.cuda.empty_cache()
        print("-----第 {} 轮训练开始-----".format(i+1))
        #batch
        for x_train, x_deltac,y_train in train_dataloader:

            x_train=torch.tensor(x_train,dtype=torch.float32,device=device)/255. #dtype -float format 转成float类型
            x_deltac=torch.tensor(x_deltac,dtype=torch.float32,device=device)
            y_train=torch.tensor(y_train,dtype=torch.float32,device=device)/255.
           # print("x_train_type",x_train.dtype)
            #print("x_deltac",x_deltac.dtype)
            #print("y_Train",y_train.dtype)
            outputs = model(x_train,x_deltac)
            #print("ytrain",y_train.shape)
           # print("output",outputs.shape)
            loss = loss_fn(outputs, y_train) # 计算实际输出与目标输出的差距
   
            # 优化器对模型调优
            optimizer.zero_grad()  # 梯度清零
            loss.backward() # 反向传播，计算损失函数的梯度
            optimizer.step()   # 根据梯度，对网络的参数进行调优
            
            total_train_step = total_train_step + 1
            #print("训练次数：{}，Loss：{}".format(total_train_step,loss))  # 方式一：获得loss值
            print("训练次数：{}Loss:{}".format(total_train_step,loss.item())) 

        
        #save
        if i% save_model_every_epoch==0:
            state={'epoch':i+1,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict()}
            torch.save(state,save_path+str(i)+'.pt')
        #print('save model')
    
     

    
    
if __name__=='__main__':
    train()
