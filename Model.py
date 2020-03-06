from keras.models import Sequential, Model, load_model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, Flatten
from keras.layers import add, dot, concatenate
from keras.layers import LSTM, Bidirectional
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras_self_attention import SeqSelfAttention
import os


# In[2]:


# Model
def SeMemNN(max_sentence_length,len_vocab,num_classes, Mem_size, Dropout_rate=0.3, data_type='float16'):
    # placeholders
    des = Input((max_sentence_length,), dtype=data_type)
    title = Input((max_sentence_length,), dtype=data_type)

    # encoders
    # embed the des sequence into a sequence of vectors
    input_encoder_m = Sequential()
    input_encoder_m.add(Embedding(input_dim=len_vocab,
                                  output_dim=Mem_size,
    #                               embeddings_initializer=my_init
                                 ))
    input_encoder_m.add(Dropout(Dropout_rate))
    # output: (samples, max_len_tra_des_sent, embedding_dim)

    input_encoder_c = Sequential()
    input_encoder_c.add(Embedding(input_dim=len_vocab,
                                  output_dim=max_sentence_length,
    #                              embeddings_initializer=my_init
                                 ))
    input_encoder_c.add(Dropout(Dropout_rate))
    # output: (samples, max_len_tra_des_sent, max_len_tra_title_sent)

    # embed the title into a sequence of vectors
    title_encoder = Sequential()
    title_encoder.add(Embedding(input_dim=len_vocab,
                                   output_dim=Mem_size,
                                   input_length=max_sentence_length,
    #                             embeddings_initializer=my_init
                               ))
    title_encoder.add(Dropout(Dropout_rate))
    # output: (samples, max_len_tra_title_sent, embedding_dim)


    input_encoded_m = input_encoder_m(des)
    input_encoded_c = input_encoder_c(title)
    title_encoded = title_encoder(title)


    # shape: (samples, max_len_tra_des_sent, max_len_tra_title_sent)
    match = dot([input_encoded_m, title_encoded], axes=(2, 2))
    match = Activation('softmax')(match)

    response = add([match, input_encoded_c])  # (samples, max_len_tra_des_sent, max_len_tra_title_sent)
    response = Permute((2, 1))(response)  # (samples, max_len_tra_title_sent, max_len_tra_des_sent)

    R = concatenate([response, title_encoded])
    R = Bidirectional(LSTM(256,return_sequences=True,dropout=Dropout_rate))(R)
    R = SeqSelfAttention(attention_width=16, attention_activation='sigmoid')(R)
    R = Flatten()(R)
    result = Dense(num_classes)(R)  # (samples, vocab_size)
    result = Activation('softmax')(result)
    model = Model([des, title], result)
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


