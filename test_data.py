from data import data_gen,load_data,w2idx,data_idx
import numpy as np



train_size=0.9
attr=2
batch_size=25
revs, W, W2, word_idx_map, vocab, mairesse ,charged_words=load_data(attr)
datasets = w2idx(revs, word_idx_map, mairesse, charged_words, attr, max_l=149, max_s=312, filter_h=3)
_D=len(datasets[0])
_S=len(datasets[0][0])
_W=len(datasets[0][0][0])
_E=W.shape[0]
dataset_idx=data_idx(attr,len(datasets[0]),batch_size)

#split train val
n_train_items=int(np.round(train_size*_D))
train_idx=dataset_idx[:n_train_items]
train_generator=data_gen(attr,train_idx,datasets,W,batch_size=25)
[train_set_x,train_set_m],train_set_y = next(train_generator)

print(train_set_x.shape)
