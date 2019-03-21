from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Activation ,Conv2D,MaxPooling2D , Input,Maximum


def BigFiveCnnModel(filter_shapes,pool_sizes,input_shape=(312, 153, 300),filter_hs=[1,2,3],hidden_units=[200,200,2],conv_non_linear='relu'):
    """
    
    """
    feature_maps = hidden_units[0] #nb FM
    model_input_layer= input(input_shape)
    layers=[]
    for i in range(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        layer=Conv2D(filters=feature_maps,kernel_size=filter_shape,activation=conv_non_linear)(model_input_layer)
        layers.append(MaxPooling2D(pool_size=pool_size)(layer))
    concat_layer=Maximum()(layers)

    
    return Model(inputs=model_input_layer,outputs=concat_layer)
