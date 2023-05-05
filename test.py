import torch
from utils import *
from config import *

def test(config):
    dataset=Dataset(config=config,type='test')
    loader = data.DataLoader(dataset,batch_size=100,collate_fn=collate_fn)
    with torch.no_grad():
        model = torch.load(config.BEST_MODEL+'best_model.pth',map_location=config.device)
        y_true_list=[]
        y_pred_list=[]

        for b,(input,target,mask) in enumerate(loader):
            input, target, mask = input.to(device=config.device), target.to(device=config.device), mask.to(device=config.device)
            y_pred = model(input,mask)
            # print(type(y_pred))
            loss = model.loss_fn(input,target,mask)
            # print('>>batch:',b,'loss',loss.item())
            for lst in y_pred:
                y_pred_list+=lst
            for y,m in zip(target,mask):
                y_true_list+=y[m==True].tolist()
            if b % 10 ==0:
                print('>>batch:',b,'loss',loss.item())
        # 整体的准确率P
        y_true_tensor = torch.tensor(y_true_list)
        y_pred_tensor = torch.tensor(y_pred_list)
        Precision = (y_true_tensor==y_pred_tensor).sum()/len(y_true_list)# y_true_tensor,就是所有预测对的/所有项目。

        # 整体的召回率R
        Recall = (y_true_tensor==y_pred_tensor).sum()/len(y_pred_list)

        # 整体的F1值
        F1 = (2*Precision*Recall)/(Precision+Recall)
        print('>> total:', len(y_true_tensor), '准确率Precision:', Precision.item(),'召回率Recall:',Recall.item(),'F1值：',F1.item())

if __name__ == '__main__':
    config = Config()
    test(config)