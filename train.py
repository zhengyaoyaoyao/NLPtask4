import torch

from utils import *
from model import *
from config import *
from tqdm import tqdm
def train(config):
    dataset = Dataset(config)
    loader = data.DataLoader(
        dataset=dataset,
        batch_size=100,
        shuffle=True,
        collate_fn=collate_fn,
    )
    model = Model(config).to(device=config.device)
    optimizer = torch.optim.Adam(model.parameters(),lr=config.LR)
    best_loss= 100
    cur_loss=0
    cur_length=0
    for epoch in range(config.EPOCH):
        for b,(input,target,mask) in enumerate(loader):
            input,target,mask = input.to(device=config.device),target.to(device=config.device),mask.to(device=config.device)
            y_pred = model(input,mask)
            loss = model.loss_fn(input,target,mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if b %50 ==0:
                print('>> epoch :',epoch,'num_b:',b,'loss:',loss.item())
            cur_loss+=loss.item()
            cur_length=b
        if cur_loss/cur_length<best_loss:
            best_loss=cur_loss/cur_length
            torch.save(model,config.BEST_MODEL+'best_model.pth')
        torch.save(model,config.MODEL_DIR+f'model_{epoch}.pth')

if __name__ == '__main__':
    config = Config()
    train(config)