from model import BigFiveCnnModel
from data import data_generator
from tensorflow.keras import backend as K
import tensorflow as tf

def nll1(y_true, y_pred):
    """ Negative log likelihood. """

    # keras.losses.binary_crossentropy give the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

def nll2(y_true, y_pred):
    """ Negative log likelihood. """

    likelihood = tf.distributions.Bernoulli(probs=y_pred)

    return - K.sum(likelihood.log_prob(y_true), axis=-1)

def train(attr=2):
    #data generator
    #load first mini-batch
    """
    train_set_x (45, 312, 153, 300)
    """
    #Model config 
    D=45#train_set_x.shape[0]
    S=312#train_set_x.shape[1]
    W=153#train_set_x.shape[2]
    E=300#train_set_x.shape[3]
    input_shape=(W,E,1)
    docs_size=S
    hidden_units=[200,200,2]
    filter_hs=[1,2,3]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((filter_h, E))
        pool_sizes.append((W-filter_h+1,1))
    model=BigFiveCnnModel(filter_shapes,pool_sizes,input_shape=input_shape,filter_hs=filter_hs,hidden_units=hidden_units,docs_size=docs_size)
    model.summary()
    model.compile(loss=nll1,optimizer="adadelta")
    train_data_generator=data_generator(attr,reshape=(W,E))
    val_data_generator=data_generator(attr,val=False,reshape=(W,E))
    model.fit_generator(generator=train_data_generator,epochs=5,validation_data=val_data_generator,steps_per_epoch=1,validation_steps=1)
    return model

    
    
    
        
    
train()

