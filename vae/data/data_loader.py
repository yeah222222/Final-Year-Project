import os
import math
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
class data_loader(Dataset):
    def __init__(self,data_dir,batch_size) :
        super(data_loader,self).__init__()
        self.data_dir=data_dir
        self.batch_size=batch_size
        self.dataset=self.create_all_batches()
        
        
    def load_data(self):
        print('Load data...')
        img_paths=os.listdir(self.data_dir)
        img_paths=[self.data_dir+img_p for img_p in img_paths]
        print("len",len(img_paths))
        return img_paths
    
    def create_all_batches(self):
    
        fps=self.load_data()

        
        fps=sorted(fps)
        all_input_imgs=np.empty(shape=(3,128,128))
    
      
        for i in range(len(fps)):

            input_img=Image.open(fps[i])
            input_img_arr=np.array(input_img).transpose((2,1,0))
            
            all_input_imgs=np.append(all_input_imgs,input_img_arr,axis=0)
            
     

        all_input_imgs=all_input_imgs[3:]

        
        all_input_imgs=all_input_imgs[np.newaxis,:,:,:]
        all_input_imgs=all_input_imgs.reshape(-1,3,128,128)
  


        #填充批次中少的样本
        
        self.input_img_batches=np.empty(shape=(1,3,128,128))
       

        
        
        
        for t in range(math.ceil(all_input_imgs.shape[0]/self.batch_size)):
            start_idx=t*self.batch_size
            end_idx=(t+1)*self.batch_size
  
            batch_input_img=all_input_imgs[start_idx:end_idx]         
            
            while(len(batch_input_img)<self.batch_size):
               
                batch_input_img=np.append(batch_input_img,batch_input_img[-1][np.newaxis,:,:,:],axis=0)
             
            
    
            self.input_img_batches=np.append(self.input_img_batches,batch_input_img,axis=0)
 
        self.input_img_batches = self.input_img_batches[1:]
    
        self.input_img_batches=self.input_img_batches[np.newaxis,:,:,:,:].reshape(-1,self.batch_size,3,128,128)
    
        return self.input_img_batches


    
    
        
    def __getitem__(self,idx):
        return self.input_img_batches[idx]
    
    
    
        
    
    
        