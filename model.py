from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Activation ,Conv2D,MaxPooling2D , Input,Concatenate, Reshape,Flatten,Dense,Embedding


def SentenceLevel(embedding_matrix,filter_shapes,pool_sizes,reshape,filter_hs=[1,2,3],hidden_units=200,conv_non_linear='relu'):
    """
    Apply a convolutional filter on EACH SENTENCE : => filter (n,E) on (W,E)
    (nb_words,E,1=channel)
    """
    S,W=reshape
    #nb_words,emb_dim=input_shape
    vocab_size,embed_dim=embedding_matrix.shape
    feature_maps = hidden_units #nb FM
    model_input_layer= Input(shape=(S,W))
    reshpe_input=Reshape((S*W,))(model_input_layer)
    embedding_layer=Embedding(vocab_size,embed_dim,weights=[embedding_matrix])(reshpe_input)
    reshape_layer=Reshape((S*W,embed_dim,1))(embedding_layer)
    layers=[]
    for i in range(len(filter_hs)):
        #print('layer'+str(i))
        filter_shape = filter_shapes[i] # ({1,2,3},300) (h,w)
        pool_size = pool_sizes[i] # S*(img_h-filter_h+1),img_w-filter_w+1
        layer=Conv2D(filters=feature_maps,kernel_size=filter_shape,activation=conv_non_linear)(reshape_layer)
        print("Conv:LAYER "+str(i)+': input='+str(layer.shape)+' // output= '+str(layer.shape))
        layer=MaxPooling2D(pool_size=pool_size)(layer)
        layers.append(layer)
        print("MAXPOOL:LAYER "+str(i)+': input='+str(layer.shape)+' // output= '+str(layer.shape))
    """
    for i in range(len(filter_hs)):
        print(Model(inputs=model_input_layer,outputs=layers[i]).summary())
    """
    concat_layer=Concatenate()(layers)

    
    return Model(inputs=model_input_layer,outputs=concat_layer)

def DocumentLevel(sentlevel,hidden_units,docs_size=312):
    # batch*docs X 1 X 1 X sentVec to batch X docs X sentVec
    #print(sentlevel.output.shape[-1])
    output=Reshape((sentlevel.output.shape[-1],))(sentlevel.output)
    # list of 84 M features per doc
    m_features=Input(shape=(84,))
    output=Concatenate()([output,m_features])
    output=Dense(hidden_units[0],activation='softmax')(output)
    output=Dense(hidden_units[1],activation="softmax")(output)
    return Model(inputs=[sentlevel.input,m_features],outputs=output)

def BigFiveCnnModel(embedding_matrix,filter_shapes,pool_sizes,reshape,filter_hs=[1,2,3],hidden_units=[200,200,2],conv_non_linear='relu',docs_size=312):
    """
    input : W X E (batch = D X S)
    input_shape=(S*W,E,1)
    """
    SentModel=SentenceLevel(embedding_matrix,filter_shapes,pool_sizes,reshape,filter_hs=filter_hs,hidden_units=hidden_units[0])
    model=DocumentLevel(SentModel,hidden_units=hidden_units[1:],docs_size=docs_size)
    return model