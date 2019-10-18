from dataprocessing import *
from Model import *
import os
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


def pltfunction(acc,loss,path,name):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    epoch = range(len(acc))
    plt.xlabel('epoch')
    plt.plot(epoch,acc,"x-",label=name+'acc')
    plt.plot(epoch,loss, "+-", label=name+'loss')
    plt.grid(True)
    plt.legend(loc=1)
    plt.savefig(path+name)
    plt.show()
    
    
# some global varible
dataset_path = './ag_news_csv/'
classes_txt = dataset_path + 'classes.txt'
train_csv = dataset_path + 'train.csv'
test_csv = dataset_path + 'test.csv'
pickle_path = './Result/'

num_classes = 4
Dropout_rate = 0.3
max_sentence_length=256
memory_size=256


tra_label,tra_title,tra_des,len_vocab,_,vocab = load_dataset(train_csv,
                                                             max_sentence_length)

tes_label,tes_title, tes_des,len_vocab_tes,max_sentence_length_tes,vocab = load_dataset(test_csv,
                                                                                        max_sentence_length,
                                                                                        shuffle=False,
                                                                                        vocab=vocab)


model = EMNN_des(max_sentence_length,len_vocab,num_classes,memory_size, 
             Dropout_rate=Dropout_rate, data_type='float32')

file_path = pickle_path + 'MemNNmodel_classification_ag_des_float32' + '.h5'

callback_list = [
                    EarlyStopping(
                        monitor='loss',
                        patience=8,
                        verbose=1,
                        mode='auto'
                    ),
                    ModelCheckpoint(
                        filepath=file_path,
                        monitor='val_loss',
                        save_best_only='True',
                        verbose=1,
                        mode='auto',
                        period=1
                    ),
                    ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto')
                    ]





training = model.fit([tra_des,tra_title], 
          tra_label,
          batch_size=64,
          epochs=30,
          callbacks=callback_list,      
          validation_data=([tes_des,tes_title], tes_label)
                    )

history = training.history
acc = np.asarray(history['acc'])
loss = np.asarray(history['loss'])
pltfunction(acc,loss,pickle_path,"training_ag_des_float32")
val_loss = np.asarray(history['val_loss'])
val_acc = np.asarray(history['val_acc'])
pltfunction(val_acc,val_loss,pickle_path,"val_ag_des_float32")

acc_and_loss = np.column_stack((acc, loss, val_acc, val_loss))
save_file_blstm = pickle_path+'MemNNmodel_classification_ag_des_float32' + '.csv'
with open(save_file_blstm, 'wb'):
    np.savetxt(save_file_blstm, acc_and_loss)

_, accuracy = model.evaluate([tes_des,tes_title], 
                                 tes_label,
                                 batch_size=256, verbose=1)

print('*******************************************************')
print("Final test validation accuracy: %s" % accuracy)
print('*******************************************************')

