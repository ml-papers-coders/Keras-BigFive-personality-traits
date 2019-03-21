from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation ,Conv2D,MaxPooling2D , Input


def BigFiveCnnModel(input_shape=(),filter_hs=[1,2,3],filter_shapes,pool_sizes,hidden_units=[200,200,2],conv_non_linear='relu'):
    """
    
    """
    feature_maps = hidden_units[0] #nb FM
    model_input_layer= input(shape=input_shape)
    layers=[]
    for i in range(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        layer=Conv2D(filters=feature_maps,kernel_size=filter_shape,activation=conv_non_linear,input_shape=input_shape))
        layers.append(MaxPooling2D(pool_size=pool_size)(layer))
    for l in layers:


    return model
