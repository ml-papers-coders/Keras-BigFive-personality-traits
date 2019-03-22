from model import BigFiveCnnModel
from data import load_data

def train(attr=2):
    #data generator
    #load first mini-batch
    train_set_x,train_set_y,val_set_x,val_set_y,train_set_m,val_set_m=next(load_data(attr,cv=0))
    """
    train_set_x (45, 312, 153, 300)
    """

    S=train_set_x[0].shape
    W=train_set_x[0][0].shape    
    print(S)
    print(W)


