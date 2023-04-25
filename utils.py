import torch
import pandas as pd
from torch.utils import data
from torch.utils.data.dataset import T_co
from config import *
import csv
def get_vacob(config):
    df = pd.read_csv(config.VOCAB_PATH,sep=' ',quoting=csv.QUOTE_NONE,encoding='utf-8',names=['word','id'])
    return list(df['word']),dict(df.values)

def get_label(config):
    df = pd.read_csv(config.LABEL_PATH,sep=' ',quoting=csv.QUOTE_NONE,encoding='utf-8',names=['label','id'])
    return list(df['label']),dict(df.values)
# 构建数据集
class Dataset(data.Dataset):
    def __init__(self,config,type='train',base_len=50):
        super().__init__()
        self.base_len = base_len
        self.WORD_UNK= config.WORD_UNK
        sample_path = config.TRAIN_SAMPLE_PATH if type=='train' else config.TEST_SAMPLE_PATH
        self.df = pd.read_csv(sample_path,sep=' ',quoting=csv.QUOTE_NONE,encoding='utf-8',names=['word','label'])
        _,self.word2id = get_vacob(config)  # 词表，把每个词用一个id表示。
        _,self.label2id = get_label(config) # label表，把每个词用一个id表示。
        self.get_points()

    def get_points(self):
        self.points=[0]
        i=0
        while True:
            if i+self.base_len>=len(self.df):  # 如果句子的长度>50，说明已经切完了
                self.points.append(len(self.df))
                break
            if self.df.loc[i+self.base_len,'label']=='O':# 找到第50个字
                i += self.base_len
                self.points.append(i)
            else:
                i+=1
    def __len__(self):
        return len(self.points)-1  # 文本数据化，可以取到3搁句子， 0-50-100-150 一共是三个句子

    def __getitem__(self, index) -> T_co:
        # 有了point坐标，就切句
        df = self.df[self.points[index]:self.points[index+1]]
        word_unk_id = self.word2id[self.WORD_UNK]
        label_o_id = self.label2id['O']
        input = [self.word2id.get(w,word_unk_id) for w in df['word']] # 输出值，默认是wrod_unk_id
        target = [self.label2id.get(l,label_o_id) for l in df['label']]
        return input,target  # 所以这里的input 和target就已经是词表里面的id了。所以如果后续使用bert，所以config那里配置一个tokenizer。

def collate_fn(batch):
    batch.sort(key = lambda x:len(x[0]),reverse = True)
    max_len = len(batch[0][0])
    input=[]
    target=[]
    mask = []
    for item in batch:# item是tuple（word，lable）
        pad_len = max_len-len(item[0])
        input.append(item[0]+[0]*pad_len) #再后面补pad_len数目
        target.append(item[1]+[1]*pad_len)
        mask.append([1]*len(item[0])+[0]*pad_len)  # 是否进行了pad处理，true就是原本，false就是原本没有
    return torch.tensor(input),torch.tensor(target),torch.tensor(mask).bool()

