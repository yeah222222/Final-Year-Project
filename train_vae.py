# import sys
# sys.path.append('/mnt/d/junch_data/test_junch')
import torch
from data.data_loader import data_loader
# from yeah import wholemodel
from torch import nn
from VAE import VAE
from torch import optim
import logging
import torch.nn.functional as F

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def train():
    
    
    logger = get_logger('/mnt/d/junch_data/test_junch/model/vae/log/train_vae_1000epoch_bs32.log',verbosity=1, name="my_logger")
 
    logger.info('start training!')

    train_dataloader=data_loader(data_dir='/mnt/d/junch_data/test_junch/model/data/train/',batch_size=64)
   
        
    #test_dataloader=data_loader(data_dir='/mnt/d/junch_data/test_junch/model/data/test',batch_size=4)

    device=torch.device("cuda:0")
    
    
    #total_train_step = 0
    epoch = 1000
    kl_loss = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    recon_loss = lambda recon_x, x: F.binary_cross_entropy(recon_x, x, size_average=False)
    model=VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    
    
    
    save_model_every_epoch=99
    save_path='/mnt/d/junch_data/test_junch/model/vae/save_model/checkpoints/'
    #epoch

    for i in range(epoch):
        torch.cuda.empty_cache()
        print("-----第 {} 轮训练开始-----".format(i+1))
        #batch
        for x_train in train_dataloader:
            print(x_train.shape)
            x_train=torch.tensor(x_train,dtype=torch.float32,device=device)/255. #dtype -float format 转成float类型
            

            
            recon_x, mu, logvar = model(x_train)
            
            recon = recon_loss(recon_x, x_train)
            kl = kl_loss(mu, logvar)
            
            loss = recon + kl
            #train_loss += loss.item()
            loss = loss/x_train.size(0)

            # 优化器对模型调优
            optimizer.zero_grad()  # 梯度清零
            loss.backward() # 反向传播，计算损失函数的梯度
            optimizer.step()   # 根据梯度，对网络的参数进行调优

            
            logger.info('Epoch:[{}/{}]\t recon_loss={:.5f}\t kl_loss={:.5f}'.format(i , epoch, recon, kl ))


            
            #save
        if i% save_model_every_epoch==0:
            state={'epoch':i+1,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict()}
            torch.save(state,save_path+str(i)+'.pt')
        print('save model')
    
     




if __name__=='__main__':
    train()
