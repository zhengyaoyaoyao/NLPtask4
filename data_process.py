import pandas as pd
import os
from glob import glob
import random
import csv
#提取英文分词
def LitBank_generate(config):
    # 确保输出文件夹存在，如果不存在则创建它
    if not os.path.exists(config.ORIGIN_LitBank_Output):
        os.makedirs(config.ORIGIN_LitBank_Output)
    # 遍历所有以 .tsv 结尾的文件
    for file_path in glob(config.ORIGIN_LitBank_DIR + '*.tsv'):
        # 构建输出文件名
        output_file_name = os.path.splitext(os.path.basename(file_path))[0] + '_processed.txt'
        output_file_path = os.path.join(config.ORIGIN_LitBank_Output, output_file_name)

        # 打开输入和输出文件
        with open(file_path, 'r', encoding='utf-8') as input_file, \
                open(output_file_path, 'w', encoding='utf-8') as output_file:
            for line in input_file:
                # 提取前两列数据并写入输出文件
                columns = line.strip().split('\t')
                if len(columns) >= 2:
                    if(',' in columns[0]):
                        continue
                    output_file.write(columns[0] + ' ' + columns[1] + '\n')
        # 输出已处理文件的信息
        print(f'Processed {file_path} -> {output_file_path}')

# 拆分英文的训练集和测试集
def Lit_Bank_split_sample(config,test_size=0):
    files = glob(config.ORIGIN_LitBank_Output+'*.txt')
    random.seed(0)
    random.shuffle(files)
    n = int(len(files) * test_size)
    train_files = files[:n]
    test_files = files[n:]
    # 合并文件
    merge_file(train_files,config.English_TRAIN_SAMPLE_PATH)
    merge_file(test_files,config.English_TEST_SAMPLE_PATH)

def merge_file(files,target_path):
    with open(target_path,'w',encoding='utf-8',errors='ignore') as file:
        for f in files:
            text = open(f,encoding='utf-8').read()
            file.write(text)

# 文件合并
def merge_train(config):
    english_file = glob(config.English_TRAIN_SAMPLE_PATH)
    chinese_file = glob(config.Chinese_TRAIN_SAMPLE_PATH)
    target_path = config.TRAIN_SAMPLE_PATH
    merge_file(english_file,target_path)
    merge_file(chinese_file,target_path)

def merge_test(config):
    english_file = glob(config.English_TEST_SAMPLE_PATH)
    chinese_file = glob(config.Chinese_TEST_SAMPLE_PATH)
    target_path = config.TEST_SAMPLE_PATH
    merge_file(english_file, target_path)
    merge_file(chinese_file, target_path)


#生成词表
def generate_vocab(config):
    df = pd.read_csv(config.TRAIN_SAMPLE_PATH, sep=' ', quoting=csv.QUOTE_NONE,encoding='utf-8', usecols=[0], names=['word'])
    vocab_list = [config.WORD_PAD,config.WORD_UNK]+df['word'].value_counts().keys().tolist()
    vocab_dict={v:k for k,v in enumerate(vocab_list)}
    vocab = pd.DataFrame(vocab_dict.items())
    vocab.to_csv(config.VOCAB_PATH,sep=' ',header=None,index=None)


#生成label
def generate_label(config):
    df = pd.read_csv(config.TRAIN_SAMPLE_PATH,sep=' ',quoting=csv.QUOTE_NONE,encoding='utf-8',usecols=[1], names=['label'])
    label_list=df['label'].value_counts().keys().tolist()
    label_dict = {v:k for k,v in enumerate(label_list)}
    label = pd.DataFrame(label_dict.items())
    label.to_csv(config.LABEL_PATH,sep=' ',header=None,index=None)


if __name__ == '__main__':
    from config import Config
    config = Config()
    # # 生成英文分词
    # LitBank_generate(config)
    # # 生成英文训练集和测试集
    # Lit_Bank_split_sample(config, 3)
    # # 中英文文件合并
    # merge_train(config)
    # merge_test(config)
    #生成label和vocab
    generate_vocab(config)
    generate_label(config)
