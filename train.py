def load_data(): 
    rng = np.random.RandomState(3435)
    #dataset shape :[trainX, trainY, testX, testY, mTrain, mTest]
    # trainX [text/,text/sents]=> text:[[w_idx,w_idx,],[]]
    # [w_idx,w_idx,] => status
    # datasets[0][0][0] : one sent
    img_h = len(datasets[0][0][0]) # max sent_length
    filter_w = img_w #embedding of each word
    feature_maps = hidden_units[0] #nb FM
    filter_shapes = []
    pool_sizes = []
    #filter hs : n =1,2,3
    for filter_h in filter_hs:
        filter_shapes.append((filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
        #(img_h-filter_h+1) FM h_size
    parameters = [("image shape",imimg_hg_h,img_w),("filter shape",filter_shapes), ("hidden_units",hidden_units),
                  ("dropout", dropout_rate), ("batch_size",batch_size),("non_static", non_static),
                    ("learn_decay",lr_decay), ("conv_non_linear", conv_non_linear), ("non_static", non_static)
                    ,("sqr_norm_lim",sqr_norm_lim),("shuffle_batch",shuffle_batch)]
    print (parameters)
    rng = np.random.RandomState(3435)


    #data words idx to embedding
    def data_idx2embed():

    layer0_input_train = 
   
    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate
    #extra data (at random)
    from utils import mini_batches
    mini_batches_generator=mini_batches(datasets.shape[0],batch_size=batch_size)
    train_mini_batches_idx=np.asarray([])
    val_mini_batches_idx=np.asarray([])
    for t,v,_ in mini_batches_generator:
        train_mini_batches_idx=np.append(train_mini_batches_idx,t)
        val_mini_batches_idx=np.append(val_mini_batches_idx,v)
    
    #divide train set into train/val sets
    ##dataset shape :[trainX, trainY, testX, testY, mTrain, mTest]
    train_set_x=datasets[0][train_mini_batches_idx]
    val_set_x=datasets[0][val_mini_batches_idx]
    train_set_y=datasets[1][train_mini_batches_idx]
    val_set_y=datasets[1][val_mini_batches_idx]
    train_set_m=datasets[4][train_mini_batches_idx]
    val_set_m=datasets[4][train_mini_batches_idxval_mini_batches_idx]
    test_set_x = datasets[2]
    test_set_y = np.asarray(datasets[3],int)
    test_set_m = datasets[5]