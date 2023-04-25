import torch.nn as nn
from torchcrf import CRF
import torch
from config import *

class Model(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.embed =nn.Embedding(config.VOCAB_SIZE,config.EMBEDDING_DIM,config.WORD_PAD_ID).to(device=config.device)
        self.lstm = nn.LSTM(config.EMBEDDING_DIM,config.HIDDEN_SIZE,batch_first=True,bidirectional=True).to(device=config.device)
        self.linear = nn.Linear(2*config.HIDDEN_SIZE,config.TARGET_SIZE).to(device=config.device)
        self.crf = CRF(config.TARGET_SIZE,batch_first=True).to(device=config.device)

    def _get_lstm_feature(self,input):
        out = self.embed(input)
        out,_=self.lstm(out)
        return self.linear(out)
    def forward(self,input,mask):
        out = self._get_lstm_feature(input)
        return self.crf.decode(out,mask)

    def loss_fn(self,input,target,mask):
        y_pred = self._get_lstm_feature(input)
        return -self.crf.forward(y_pred,target,mask,reduction='mean')


if __name__ == '__main__':
    from config import Config
    config = Config()
    model = Model(config)
    input = torch.randint(0,3000,(100,50))
    print(len(model(input,None)))