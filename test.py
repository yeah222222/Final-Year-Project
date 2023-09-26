import torch
from data.data_loader import data_loader
# from yeah import wholemodel
from torch import nn
from whole_model import wholemodel
from PIL import Image

def test():
        
    #test_dataloader=data_loader(data_dir='/mnt/d/junch_data/test_junch/model/data/test/',batch_size=4)
    test_dataloader=data_loader(data_dir='/mnt/d/junch_data/test_junch/model/data/test/',batch_size=4)

    device=torch.device("cuda:0")   
    epoch=1
    criteria=nn.MSELoss()
    
    save_path='/mnt/d/junch_data/test_junch/model/save_model/checkpoint/'
    RESUME_EPOCH=9
    RESUME_MODEL=save_path+str(RESUME_EPOCH)+'.pt'
    model=wholemodel().to(device)
    checkpoint=torch.load(RESUME_MODEL)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.requires_grad=False

    
    for i in range(epoch):
        print("-----第 {} 轮测试开始-----".format(i+1))
        #batch
        for x_test, x_deltac,y_test in test_dataloader:

            x_test=torch.tensor(x_test,dtype=torch.float32,device=device)/255. #dtype -float format 转成float类型
            x_deltac=torch.tensor(x_deltac,dtype=torch.float32,device=device)
            y_test=torch.tensor(y_test,dtype=torch.float32,device=device)/255.

            outputs = model(x_test,x_deltac)
            img=outputs.detach().cpu().numpy()
            print(img.shape)
            for j in img:
                j=j*255.0
                print(j.shape)
                j = Image.fromarray(j.astype('uint8')).convert('RGB')
                print(j.shape)
                output_path = '/mnt/d/junch_data/test_junch/model/test_res/'+str(j)+".jpg"
                j.save(output_path)
    
            distance=criteria(y_test,outputs)
            
            print(distance)


if __name__=='__main__':
    test()