import torch
from data.data_loader import data_loader
# from yeah import wholemodel
from torch import nn
#from whole_model import wholemodel
from VAE import VAE
from PIL import Image
import numpy as np
def test():
        
    #test_dataloader=data_loader(data_dir='/mnt/d/junch_data/test_junch/model/data/test/',batch_size=4)
    test_dataloader=data_loader(data_dir='/mnt/d/junch_data/test_junch/model/data/test/',batch_size=1)
    # for x_test,_,_ in test_dataloader:
    #     print(x_test.shape)
    input_img=Image.open('/mnt/d/junch_data/test_junch/model/data/test/anise_002_normalized_3_9_142.jpg')
    x_test=np.array(input_img).transpose((2,1,0))
    x_test=x_test[np.newaxis,:,:,:].reshape(1,3,128,128)
    # print("type",type(x_test))
    # print("x_test_view_size0",x_test.size[0])
    # x_test_view=x_test.view(x_test.size[0],-1)
    # print(x_test.shape)
    # print(x_test_view.shape)
    # x_test=x_test[np.newaxis,:,:,:,:].reshape(1,1,3,128,128)
    device=torch.device("cuda:0")   
    epoch=1
    #criteria=nn.MSELoss()
    
    save_path='/mnt/d/junch_data/test_junch/model/vae/save_model/checkpoints/'
    RESUME_EPOCH=990
    RESUME_MODEL=save_path+str(RESUME_EPOCH)+'.pt'
    model=VAE().to(device)
    checkpoint=torch.load(RESUME_MODEL)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.requires_grad=False

    
    for i in range(epoch):
        print("-----第 {} 轮测试开始-----".format(i+1))
        #batch
        
        x_test=torch.tensor(x_test,dtype=torch.float32,device=device)/255. #dtype -float format 转成float类型
        # x_deltac=torch.tensor(x_deltac,dtype=torch.float32,device=device)
        
        #  y_test=torch.tensor(y_test,dtype=torch.float32,device=device)/255.

        outputs = model(x_test)
        outputs=outputs[0]
        img=outputs.detach().cpu().numpy().squeeze()    
        img=np.transpose(img,(1,2,0))
        img=img*255.0
        img=img
        imges = Image.fromarray(img.astype('uint8')).convert('RGB')
        output_path = '/mnt/d/junch_data/test_junch/model/vae/test_res_VAE/'+"train222ywwweah"+str(RESUME_EPOCH)+".jpg"
        imges.save(output_path)
            #distance=criteria(y_test,outputs)
            
            #print(distance)


if __name__=='__main__':
    test()