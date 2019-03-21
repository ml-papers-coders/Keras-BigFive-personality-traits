
import pickle as cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation ,Conv2D,MaxPooling2D
#import theano
#print('Theano loaded')
#import theano.tensor as T
#from theano.ifelse import ifelse
import os
import warnings
import sys
import time
import getpass
import csv
warnings.filterwarnings("ignore")



def train_conv_net(datasets,
                   U,
                   ofile,
                   cv=0,
                   attr=0,
                   img_w=300,
                   filter_hs=[3,4,5],# n in [1,2,3]
                   hidden_units=[100,2], #[200,200,2]
                   dropout_rate=[0.5],
                   shuffle_batch=True,
                   n_epochs=25,
                   batch_size=25,
                   lr_decay = 0.95,
                   conv_non_linear="relu",
                   activations=[Iden],
                   sqr_norm_lim=9,
                   non_static=True):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """
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
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
        #(img_h-filter_h+1) FM h_size
    parameters = [("image shape",img_h,img_w),("filter shape",filter_shapes), ("hidden_units",hidden_units),
                  ("dropout", dropout_rate), ("batch_size",batch_size),("non_static", non_static),
                    ("learn_decay",lr_decay), ("conv_non_linear", conv_non_linear), ("non_static", non_static)
                    ,("sqr_norm_lim",sqr_norm_lim),("shuffle_batch",shuffle_batch)]
    print (parameters)

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
    val_set_m=datasets[4][val_mini_batches_idx]
    test_set_x = datasets[2]
    test_set_y = np.asarray(datasets[3],int)
    test_set_m = datasets[5]
    

    conv_layers = Sequential()
    
    #just define the conv layers for the the different n values
    #sentence embedding
    for i in range(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layers.add(Conv2D(filter_shape,activation=conv_non_linear,input_shape=train_set_x.shape[1:]))
        conv_layers.add(MaxPooling2D(pool_size=pool_size))
        #conv_layer = LeNetConvPoolLayer(rng, image_shape=None,
        #                        filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        #conv_layers.append(conv_layer)

    # ???

    def convolve_user_statuses(statuses):
        layer1_inputs = []
        ##???
        def sum_mat(mat, out):
            z=ifelse(T.neq(T.sum(mat),T.constant(0)),T.constant(1),T.constant(0))
            return  out+z, theano.scan_module.until(T.eq(z,T.constant(0)))

        status_count,_ = theano.scan(fn = sum_mat, sequences=statuses, outputs_info=T.constant(0,dtype=theano.config.floatX))

        # Slice-out dummy (zeroed) sentences
        relv_input=statuses[:T.cast(status_count[-1],dtype='int32')].dimshuffle(0, 'x', 1, 2)

        for conv_layer in conv_layers:
            layer1_inputs.append(conv_layer.convolve(input=relv_input).flatten(2))

        features = T.concatenate(layer1_inputs, axis=1)

        avg_feat = T.max(features, axis=0)

        return avg_feat

    conv_feats, _ = theano.scan(fn= convolve_user_statuses, sequences= layer0_input)

    # Add Mairesse features
    layer1_input = T.concatenate([conv_feats, mair], axis=1)##mairesse_change
    hidden_units[0] = feature_maps*len(filter_hs) + datasets[4].shape[1]##mairesse_change
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)

    svm_data = T.concatenate([classifier.layers[0].output, y.dimshuffle(0, 'x')], axis = 1)
    #define parameters of the model and update functions using adadelta
    params = classifier.params
    for conv_layer in conv_layers:
        params += conv_layer.params
    if non_static:
        #if word vectors are allowed to change, add them as model parameters
        params += [idx_embed]
    cost = classifier.negative_log_likelihood(y)
    dropout_cost = classifier.dropout_negative_log_likelihood(y)
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)


    test_y_pred = classifier.predict(layer1_input)
    test_error = T.sum(T.neq(test_y_pred, y))
    true_p = T.sum(test_y_pred*y)
    false_p = T.sum(test_y_pred*T.mod(y+T.ones_like(y),T.constant(2,dtype='int32')))
    false_n = T.sum(y*T.mod(test_y_pred+T.ones_like(y),T.constant(2,dtype='int32')))
    test_model_all = theano.function([x, y,
                                        mair##mairesse_change
                                        ]
                                    , [test_error, true_p, false_p, false_n, svm_data], allow_input_downcast = True)

    test_batches = test_set_x.shape[0]/batch_size;


    #start training over mini-batches
    print ('... training')
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    test_perf = 0
    fscore = 0
    cost_epoch = 0
    while (epoch < n_epochs):
        start_time = time.time()
        epoch = epoch + 1
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        train_losses = [test_model(i) for i in xrange(n_train_batches)]
        train_perf = 1 - np.mean([loss[0] for loss in train_losses])
        val_losses = [val_model(i) for i in xrange(n_val_batches)]
        val_perf = 1- np.mean(val_losses)
        epoch_perf = 'epoch: %i, training time: %.2f secs, train perf: %.2f %%, val perf: %.2f %%' % (epoch, time.time()-start_time, train_perf * 100., val_perf*100.)
        print(epoch_perf)
        ofile.write(epoch_perf+"\n")
        ofile.flush()
        if val_perf >= best_val_perf:
            best_val_perf = val_perf
            test_loss_list = [test_model_all(test_set_x[idx*batch_size:(idx+1)*batch_size], test_set_y[idx*batch_size:(idx+1)*batch_size],
            test_set_m[idx*batch_size:(idx+1)*batch_size]##mairesse_change
            ) for idx in xrange(test_batches)]
            if test_set_x.shape[0]>test_batches*batch_size:
                test_loss_list.append(test_model_all(test_set_x[test_batches*batch_size:], test_set_y[test_batches*batch_size:],
                test_set_m[test_batches*batch_size:]##mairesse_change
                ))
            test_loss_list_temp=test_loss_list
            test_loss_list=np.asarray([t[:-1] for t in test_loss_list])
            test_loss = np.sum(test_loss_list[:, 0])/float(test_set_x.shape[0])
            test_perf = 1- test_loss
            tp = np.sum(test_loss_list[:, 1])
            fp = np.sum(test_loss_list[:, 2])
            fn = np.sum(test_loss_list[:, 3])
            tn = test_set_x.shape[0]-(tp+fp+fn)
            fscore=np.mean([2*tp/float(2*tp+fp+fn), 2*tn/float(2*tn+fp+fn)])
            svm_test=np.concatenate([t[-1] for t in test_loss_list_temp], axis=0)
            svm_train=np.concatenate([t[1] for t in train_losses], axis=0)
            output="Test result: accu: "+str(test_perf)+", macro_fscore: "+str(fscore)+"\ntp: "+str(tp)+" tn:"+str(tn)+" fp: "+str(fp)+" fn: "+str(fn)
            print (output)
            ofile.write(output+"\n")
            ofile.flush()
            # dump train and test features
            cPickle.dump(svm_test, open("cvte"+str(attr)+str(cv)+".p", "wb"))
            cPickle.dump(svm_train, open("cvtr"+str(attr)+str(cv)+".p", "wb"))
        updated_epochs = refresh_epochs()
        if updated_epochs!=None and n_epochs!=updated_epochs:
            n_epochs = updated_epochs
            print( 'Epochs updated to '+str(n_epochs))
    return test_perf, fscore

def refresh_epochs():
    try:
        f=open('n_epochs','r')
    except Exception:
        return None

    try:
        n = int(f.readline().strip())
    except Exception:
        f.close()
        return None
    f.close()
    return n


def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y, data_m = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_m = theano.shared(np.asarray(data_m,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32'), shared_m

def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='idx_embed'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        updates[param] = stepped_param
    return updates

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)

def safe_update(dict_to, dict_from):
    """
    re-make update dictionary for safe updating
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to

def get_idx_from_sent(status, word_idx_map, charged_idx_embed, max_l=51, max_s=200, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    length = len(status)


    #pass for the while not the for loop
    pass_one=True
    #random : au pire des cas on obtient un x vide alors une boucle while pour eviter ce prob
    while len(x)==0:
        for i in range(length):
            idx_embed = status[i].split()
            if pass_one:
                idx_embed_set = set(idx_embed)
                if len(charged_idx_embed.intersection(idx_embed_set))==0:
                    continue
            else:
                if np.random.randint(0,2)==0:
                    continue
            y=[]
            for i in range(pad):
                y.append(0)
            for word in idx_embed:
                if word in word_idx_map:
                    y.append(word_idx_map[word])

            while len(y) < max_l+2*pad:
                y.append(0)
            x.append(y)
        #we are here because of our bad luck : len(x)==0 no need to set the idx_embed
        pass_one=False

    if len(x) < max_s:
        #list extend :http://thomas-cokelaer.info/blog/2011/03/post-2/
        x.extend([[0]*(max_l+2*pad)]*(max_s-len(x)))
    return x

def make_idx_data_cv(revs, word_idx_map, mairesse, charged_idx_embed, cv, per_attr=0, max_l=51, max_s=200, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    trainX, testX, trainY, testY, mTrain, mTest = [], [], [], [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map,
        charged_idx_embed,
        max_l, max_s, k, filter_h)

        if rev["split"]==cv:
            testX.append(sent)
            testY.append(rev['y'+str(per_attr)])
            mTest.append(mairesse[rev["user"]])
        else:
            trainX.append(sent)
            trainY.append(rev['y'+str(per_attr)])
            mTrain.append(mairesse[rev["user"]])
    trainX = np.array(trainX,dtype=int)
    testX = np.array(testX,dtype=int)
    trainY = np.array(trainY,dtype=int)
    testY = np.array(testY,dtype=int)
    #mTrain is the mairesse features
    mTrain = np.array(mTrain, dtype=float)
    mTest = np.array(mTest, dtype=float)
    return [trainX, trainY, testX, testY, mTrain, mTest]


if __name__=="__main__":
    print ("loading data...")
    x = cPickle.load(open("processed.pkl","rb"))
    revs, W, W2, word_idx_map, vocab, mairesse = x[0], x[1], x[2], x[3], x[4], x[5]
    print ("data loaded!")
    mode= sys.argv[1]
    word_vectors = sys.argv[2]
    attr = int(sys.argv[3])
    if mode=="-nonstatic":
        print ("model architecture: CNN-non-static")
        non_static=True
    elif mode=="-static":
        print ("model architecture: CNN-static")
        non_static=False
    #execfile("conv_net_classes.py")
    exec(open("./conv_net_classes.py").read())
    if word_vectors=="-rand":
        print ("using: random vectors")
        U = W2
    elif word_vectors=="-word2vec":
        print ("using: word2vec vectors")
        U = W
    ###################################################
    #outputfile :
    ofile=open('perf_output_'+str(attr)+'.txt','w')
    #list of idx_embed when the charged is 1
    charged_idx_embed=[]

    emof=open("Emotion_Lexicon.csv","rb")
    csvf=csv.reader(emof, delimiter=',',quotechar='"')
    first_line=True

    for line in csvf:
        if first_line:
            first_line=False
            continue
        if line[11]=="1":
            charged_idx_embed.append(line[0])

    emof.close()

    charged_idx_embed=set(charged_idx_embed)

    results = []
    #cv fold
    r = range(0,10)
    for i in r:
        #datasets:[trainX, trainY, testX, testY, mTrain, mTest]
        datasets = make_idx_data_cv(revs, word_idx_map, mairesse, charged_idx_embed, i, attr, max_l=149, max_s=312, k=300, filter_h=3)

        perf, fscore = train_conv_net(datasets,
                              U,
                              ofile,
                              cv=i,
                              attr=attr,
                              lr_decay=0.95,
                              filter_hs=[1,2,3],
                              conv_non_linear="relu",
                              hidden_units=[200,200,2],
                              shuffle_batch=True,
                              n_epochs=50,
                              sqr_norm_lim=9,
                              non_static=non_static,
                              batch_size=50,
                              dropout_rate=[0.5, 0.5, 0.5],
                              activations=[Sigmoid])
        output = "cv: " + str(i) + ", perf: " + str(perf)+ ", macro_fscore: " + str(fscore)
        print (output)
        ofile.write(output+"\n")
        ofile.flush()
        results.append([perf, fscore])
    results=np.asarray(results)
    perf_out = 'Perf : '+str(np.mean(results[:, 0]))
    fscore_out = 'Macro_Fscore : '+str(np.mean(results[:, 1]))
    print (perf_out)
    print (fscore_out)
    ofile.write(perf_out+"\n"+fscore_out)
    ofile.close()
