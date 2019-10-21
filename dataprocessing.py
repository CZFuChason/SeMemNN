
import numpy as np
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary


def split_sent(instance):
    return instance['description'].split()

def token(dataset,max_sentence_length=None,vocab=None):
    def padding_words(data):
        for i in range(len(data)):
            if data[i]['description_seq_len'] <= max_sentence_length:
                padding = [-1] * (max_sentence_length - data[i]['description_seq_len'])
                data[i]['description_words'] += padding
#                 for _ in range(max_sentence_length - data[i]['description_seq_len']):
#                     data[i]['description_words'].append(-1)
            else:
                pass
        for i in range(len(data)):
            if data[i]['title_seq_len'] <= max_sentence_length:
                padding = [-1] * (max_sentence_length - data[i]['title_seq_len'])
                data[i]['title_words'] += padding
#                 for _ in range(max_sentence_length - data[i]['title_seq_len']):
#                     data[i]['title_seq_len'].append(-1)
            else:
                pass
        return data

    max_des_len_train=0
    max_title_len_train=0

    for i in range (len(dataset)):
        if(dataset[i]['description_seq_len'] > max_des_len_train):
            max_des_len_train = dataset[i]['description_seq_len']
        else:
            pass
    for i in range (len(dataset)):
        if(dataset[i]['title_seq_len'] > max_title_len_train):
            max_title_len_train = dataset[i]['title_seq_len']
        else:
            pass
    if max_sentence_length==None:
        max_sentence_length = max_des_len_train
        if (max_title_len_train > max_sentence_length):
            max_sentence_length = max_des_len_train
    print ('max_sentence_length:',max_sentence_length)
    
    if vocab==None:
        print('New Vocab')
        vocab = Vocabulary(min_freq=2)
        dataset.apply(lambda x:[vocab.add(word) for word in x['description_words']])
        dataset.apply(lambda x:[vocab.add(word) for word in x['title_words']])
        vocab.build_vocab()
        
        dataset.apply(lambda x: [vocab.to_index(word) for word in x['description_words']],
                      new_field_name='description_words')
        dataset.apply(lambda x: [vocab.to_index(word) for word in x['title_words']],
                      new_field_name='title_words')
        dataset= padding_words(dataset)
    else:
        print('Vocab exist')
        dataset.apply(lambda x: [vocab.to_index(word) for word in x['description_words']],
                      new_field_name='description_words')
        dataset.apply(lambda x: [vocab.to_index(word) for word in x['title_words']],
                      new_field_name='title_words')
        dataset= padding_words(dataset)
        
    return dataset,len(vocab),max_sentence_length,vocab


def load_file(file_path, max_len,vocab):
    # read csv data to DataSet
    dataset = DataSet.read_csv(file_path,headers=('label','title','description'),sep='","')
    # preprocess data
    dataset.apply(lambda x: int(x['label'][1])-1,new_field_name='label')
#     dataset.apply(lambda x: int(x['label'][0])-1,new_field_name='label')
    dataset.apply(lambda x: x['title'].lower(), new_field_name='title')
    dataset.apply(lambda x: x['description'][:-2].lower()+' .', new_field_name='description')
    dataset.apply(split_sent,new_field_name='description_words')
    dataset.apply(lambda x: len(x['description_words']),new_field_name='description_seq_len')
    dataset.apply(split_sent,new_field_name='title_words')
    dataset.apply(lambda x: len(x['title_words']),new_field_name='title_seq_len')
    dataset,len_vocab,max_sentence_length,vocab = token(dataset,max_len,vocab)
    return dataset,len_vocab,max_sentence_length,vocab


def tolist(dataset,shuffle=True):
    tra_data_list=[]
    for data in dataset:
        tra_data_list.append([data['label'],data['title_words'],data['description_words']])
    if shuffle:
        np.random.shuffle(tra_data_list)
    return tra_data_list


def data_split(dataset,shuffle=True):
    dataset=tolist(dataset,shuffle)
    label, title, description = [], [], []
    for data in dataset:
        label.append(data[0])
        title.append(data[1])
        description.append(data[2])
    return [label,title,description]


def load_dataset(path,max_len=None,shuffle=True,vocab=None):
    dataset,len_vocab,max_sentence_length, vocab = load_file(path,max_len,vocab)
    [label,title,description]=data_split(dataset,shuffle=True)
    return label,title,description,len_vocab,max_sentence_length,vocab

