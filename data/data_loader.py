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
        #self.test_dir=test_dir
        self.batch_size=batch_size
        # self.stage=stage
        self.dataset=self.create_all_batches()
        
        
    def load_data(self):
        print('Load data...')
        img_paths=os.listdir(self.data_dir)
        img_paths=[self.data_dir+img_p for img_p in img_paths]
        return img_paths
    
    def create_all_batches(self):#fps: filepath for all the input images fps中可能包含不同类别的，后续要改一下 ，先假设都是同一个3D物体
    
        fps=self.load_data()

        
        fps=sorted(fps)
        all_input_imgs=np.empty(shape=(3,128,128))
        all_deltacs=np.empty(shape=(3))
        all_target_imgs=np.empty(shape=(3,128,128))
        
        #calculate deltac
      
        for i in range(len(fps)-1):
            for j in range(i+1,len(fps)):

                if fps[i].split("/")[-1].split(".")[0].split("_")[1]==fps[j].split("/")[-1].split(".")[0].split("_")[1]:                    
                    deltac=self.cal_deltac(fps[i],fps[j])
                    input_img=Image.open(fps[i])
                    input_img_arr=np.array(input_img).transpose((2,1,0))
                    output_img=Image.open(fps[j])
                    output_img_arr=np.array(output_img).transpose((2,1,0))

                    all_input_imgs=np.append(all_input_imgs,input_img_arr,axis=0)
                    
                    all_deltacs=np.append(all_deltacs,deltac,axis=0)
                    all_target_imgs=np.append(all_target_imgs,output_img_arr,axis=0)
                else:
                    continue
        all_input_imgs=all_input_imgs[3:]
        all_deltacs=all_deltacs[3:]
        all_target_imgs=all_target_imgs[3:]
        
        all_input_imgs=all_input_imgs[np.newaxis,:,:,:]
        all_input_imgs=all_input_imgs.reshape(-1,3,128,128)
        all_deltacs=all_deltacs[np.newaxis,:]
        all_deltacs=all_deltacs.reshape(-1,3)
        all_target_imgs=all_target_imgs[np.newaxis,:,:,:]
        all_target_imgs=all_target_imgs.reshape(-1,3,128,128)


        #填充批次中少的样本
        
        self.input_img_batches=np.empty(shape=(1,3,128,128))
        self.deltac_batches=np.empty(shape=(1,3))
        self.out_target_batches=np.empty(shape=(1,3,128,128))
        
        
        
        for t in range(math.ceil(all_input_imgs.shape[0]/self.batch_size)):
            start_idx=t*self.batch_size
            end_idx=(t+1)*self.batch_size
         
            #print(all_input_imgs)
            batch_input_img=all_input_imgs[start_idx:end_idx]
            batch_deltac=all_deltacs[start_idx:end_idx]
            batch_target_img=all_target_imgs[start_idx:end_idx]
            
            
            
            while(len(batch_input_img)<self.batch_size):
                #self.input_img_batches.append(batch_input_img[-1])
                batch_input_img=np.append(batch_input_img,batch_input_img[-1][np.newaxis,:,:,:],axis=0)
                batch_deltac=np.append(batch_deltac,batch_deltac[-1][np.newaxis,:],axis=0)
                batch_target_img=np.append(batch_target_img,batch_target_img[-1][np.newaxis,:,:,:],axis=0)
            
    
            self.input_img_batches=np.append(self.input_img_batches,batch_input_img,axis=0)
            self.deltac_batches=np.append(self.deltac_batches,batch_deltac,axis=0)
            self.out_target_batches=np.append(self.out_target_batches,batch_target_img,axis=0)


        # print("input img batches",self.input_img_batches)
        
        #         all_input_imgs=all_input_imgs[3:]
        # all_deltacs=all_deltacs[3:]
        # all_target_imgs=all_target_imgs[3:]
 
        self.input_img_batches = self.input_img_batches[1:]
        self.deltac_batches= self.deltac_batches[1:]
        self.out_target_batches= self.out_target_batches[1:]
        
        self.input_img_batches=self.input_img_batches[np.newaxis,:,:,:,:].reshape(-1,self.batch_size,3,128,128)
        self.deltac_batches=self.deltac_batches[np.newaxis,:,:].reshape(-1,self.batch_size,3)
        self.out_target_batches=self.out_target_batches[np.newaxis,:,:,:,:].reshape(-1,self.batch_size,3,128,128)
        

        return self.input_img_batches,self.deltac_batches,self.out_target_batches

    def cal_deltac(self,fp1,fp2):#filepath1 filepath2
        d1=int(fp1.split("/")[-1].split(".")[0].split("_")[-3])
        e1=int(fp1.split("/")[-1].split(".")[0].split("_")[-2])
        a1=int(fp1.split("/")[-1].split(".")[0].split("_")[-1])
        
        d2=int(fp2.split("/")[-1].split(".")[0].split("_")[-3])
        e2=int(fp2.split("/")[-1].split(".")[0].split("_")[-2])
        a2=int(fp2.split("/")[-1].split(".")[0].split("_")[-1])
        
        
        deltac=(d2-d1,e2-e1,a2-a1)
        
        return deltac
    
    # def load_test_data(self):
    #     print('Load test data...')
    #     img_paths=os.listdir(self.test_dir)
    #     img_paths=[self.test_dir+img_p for img_p in img_paths]
    #     return img_paths
    
    
    def __getitem__(self,idx):
        return self.input_img_batches[idx],self.deltac_batches[idx],self.out_target_batches[idx]
    
    
    
        
    
    
        