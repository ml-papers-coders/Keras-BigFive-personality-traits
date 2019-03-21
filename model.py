from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Activation ,Conv2D,MaxPooling2D , Input,Concatenate, Reshape,Flatten


def SentenceLevel(filter_shapes,pool_sizes,input_shape=(153,300,1),filter_hs=[1,2,3],hidden_units=[200,200,2],conv_non_linear='relu'):
    """
    Apply a convolutional filter on EACH SENTENCE : => filter (n,E) on (W,E)
    (nb_words,E,1=channel)
    """
    #nb_words,emb_dim=input_shape
    feature_maps = hidden_units[0] #nb FM
    model_input_layer= Input(shape=(input_shape))
    layers=[]
    for i in range(len(filter_hs)):
        #print('layer'+str(i))
        filter_shape = filter_shapes[i] # ({1,2,3},300) (h,w)
        pool_size = pool_sizes[i] # img_h-filter_h+1,img_w-filter_w+1
        layer=Conv2D(filters=feature_maps,kernel_size=filter_shape,activation=conv_non_linear)(model_input_layer)
        layer=MaxPooling2D(pool_size=pool_size)(layer)
        layers.append(layer)
    """
    for i in range(len(filter_hs)):
        print(Model(inputs=model_input_layer,outputs=layers[i]).summary())
    """
    concat_layer=Concatenate()(layers)

    
    return model_input_layer,concat_layer

def DocumentLevel(sentlevel_output,docs_size):
    # batch*docs X 1 X 1 X sentVec to batch X docs X sentVec
    output=Reshape((docs_size,sentlevel_output.shape[-1]))(sentlevel_output)
    # list of 84 M features per doc
    m_features=Input(shape=())
    return output
