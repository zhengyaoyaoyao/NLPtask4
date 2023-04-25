import torch
from utils import *
from model import *
from config import *

def test(config):
    dataset=Dataset(config=config,type='test')
    loader = data.DataLoader(dataset,batch_size=100,collate_fn=collate_fn)
    with torch.no_grad():
        model = torch.load(config.MODEL_DIR+'(cuda)model_0.pth')
        y_true_list=[]
        y_pred_list=[]

        for b,(input,target,mask) in enumerate(loader):
            input, target, mask = input.to(device=config.device), target.to(device=config.device), mask.to(device=config.device)
            y_pred = model(input,mask)
            print(type(y_pred))
            loss = model.loss_fn(input,target,mask)
            print('>>batch:',b,'loss',loss.item())

            for lst in y_pred:
                y_pred_list+=lst
            for y,m in zip(target,mask):
                y_true_list+=y[m==True].tolist()

        # 整体的准确率
        y_true_tensor = torch.tensor(y_true_list)
        y_pred_tensor = torch.tensor(y_pred_list)
        accuracy = (y_true_tensor==y_pred_tensor).sum()/len(y_true_tensor)
        print('>> total:', len(y_true_tensor), 'accuracy:', accuracy.item())
