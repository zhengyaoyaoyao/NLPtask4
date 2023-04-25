import torch
class Config:
    def __init__(self):
        # 英文数据集
        self.ORIGIN_LitBank_DIR='./litbank/entities/tsv/'
        self.ORIGIN_LitBank_Output='./litbank/output/'
        self.English_TRAIN_SAMPLE_PATH='./litbank/train.txt'
        self.English_TEST_SAMPLE_PATH='./litbank/test.txt'


        # 中文数据集
        self.Chinese_TRAIN_SAMPLE_PATH = './Chinese-Literature-NER-RE-Dataset/ner/train.txt'
        self.Chinese_TEST_SAMPLE_PATH='./Chinese-Literature-NER-RE-Dataset/ner/test.txt'
        self.Chinese_VALIDATION_SAMPLE_PATH='./Chinese-Literature-NER-RE-Dataset/ner/validation.txt'

        # 总的文件
        self.TEST_SAMPLE_PATH='./output/test.txt'
        self.TRAIN_SAMPLE_PATH='./output/train.txt'
        self.VOCAB_PATH='./output/vocab.txt'
        self.LABEL_PATH='./output/label.txt'

        self.WORD_PAD='<PAD>'
        self.WORD_UNK='<UNK>'
        self.WORD_PAD_ID = 0
        self.WORD_UNK_ID = 1
        self.LABEL_O_ID = 0
        self.VOCAB_SIZE = 22903

        self.EMBEDDING_DIM = 100
        self.HIDDEN_SIZE = 256
        self.TARGET_SIZE = 33
        self.LR=1e-4 # 学习率
        self.EPOCH=100  # 训练轮次
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.MODEL_DIR='./output/model/'
        self.BEST_MODEL='./output/bestModel/'

if __name__ == '__main__':
    print(torch.cuda.is_available())