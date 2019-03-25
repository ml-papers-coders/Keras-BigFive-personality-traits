from model import BigFiveCnnModel
from tensorflow.keras import backend as K
import tensorflow as tf
from data import data_idx,load_data,w2idx,data_gen
import numpy as np


def nll1(y_true, y_pred):
    """ Negative log likelihood. """

    # keras.losses.binary_crossentropy give the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

def nll2(y_true, y_pred):
    """ Negative log likelihood. """

    likelihood = tf.distributions.Bernoulli(probs=y_pred)

    return - K.sum(likelihood.log_prob(y_true), axis=-1)

def init(attr=2,train_size=0.9,batch_size=25):
    #data generator
    #load first mini-batch
    """
    train_set_x (45, 312, 153, 300)
    """
    
    revs, W, W2, word_idx_map, vocab, mairesse ,charged_words=load_data(attr)
    datasets = w2idx(revs, word_idx_map, mairesse, charged_words, attr, max_l=149, max_s=312, filter_h=3)
    _D=len(datasets[0])
    _S=len(datasets[0][0])
    _W=len(datasets[0][0][0])
    _E=W.shape[1]
    dataset_idx=data_idx(attr,len(datasets[0]),batch_size)

    #split train val
    n_train_items=int(np.round(train_size*_D))
    train_idx=dataset_idx[:n_train_items]
    test_idx=dataset_idx[n_train_items:]
    train_generator=data_gen(attr,train_idx,datasets,W,batch_size=25)
    test_generator=data_gen(attr,test_idx,datasets,W,batch_size=25)
    input_shape=(_S*_W,_E,1)
    docs_size=_S
    hidden_units=[200,200,2]
    filter_hs=[1,2,3]
    filter_shapes = []
    pool_sizes = []
    reshape=(_S,_W)
    for filter_h in filter_hs:
        filter_shapes.append((filter_h, _E))
        pool_sizes.append((_S*(_W-filter_h+1),1))
    model=BigFiveCnnModel(W,filter_shapes,pool_sizes,reshape,filter_hs=filter_hs,hidden_units=hidden_units,docs_size=docs_size)
    model.summary()
    model.compile(loss=nll1,optimizer="adadelta")
    steps=train_idx.shape[0]//batch_size
    return model,train_generator,test_generator,steps

    
def train(attr=2):
    model,train_generator,test_generator,steps=init(attr)
    model.fit_generator(
    generator=train_generator,
    epochs=1,
    validation_data=test_generator,
    steps_per_epoch=steps
    )


    
        
    
train()

