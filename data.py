import numpy as np 
import csv
import pickle


def get_idx_from_sent(status, word_idx_map, charged_words, max_l=51, max_s=200, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeros
    randomly some sentences are not 
    status : whole text is a list of sentences and long sentences are already splitted
    """
    x = []
    pad = filter_h - 1
    length = len(status)

    #pass for the while not the for loop
    pass_one=True
    #random : au pire des cas on obtient un x vide alors une boucle while pour eviter ce prob
    while len(x)==0:
        for i in range(length):
            words = status[i].split()
            if pass_one:
                words_set = set(words)
                if len(charged_words.intersection(words_set))==0:
                    continue
            else:
                #randomly pass same sentences
                if np.random.randint(0,2)==0:
                    continue
            y=[]
            for i in range(pad):
                y.append(0)
            for word in words:
                if word in word_idx_map:
                    y.append(word_idx_map[word])

            while len(y) < max_l+2*pad:
                y.append(0)
            x.append(y)
        #we are here because of our bad luck : len(x)==0 no need to set the words
        pass_one=False

    if len(x) < max_s:
        #list extend :http://thomas-cokelaer.info/blog/2011/03/post-2/
        x.extend([[0]*(max_l+2*pad)]*(max_s-len(x)))
    return x

def make_idx_data_cv(revs, word_idx_map, mairesse, charged_words, cv, per_attr=0, max_l=51, max_s=200, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    trainX, testX, trainY, testY, mTrain, mTest = [], [], [], [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map,
        charged_words,
        max_l, max_s, filter_h)

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

def mini_batches(data_size,batch_size,test_size=0.1,seed=3435):
    """
    return indices
    shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate
    extra data (at random)
    """
    train_size=1-test_size
    np.random.seed(seed)
    if data_size % batch_size > 0 :
        extra_data_num = batch_size - data_size % batch_size
        
        rand_perm = np.random.permutation(range(data_size))
        extra_data=rand_perm[:extra_data_num]
        
        # add random replication
        new_data=np.append(np.arange(data_size),extra_data)
    else:
        new_data=np.random.permutation(range(data_size))

    # Shuffling the new data
    n_batches = int(new_data.shape[0]/batch_size)
    rand_perm = np.random.permutation(range(len(new_data)))
    new_data=new_data[rand_perm]

    #divide train set into train/val sets
    n_train_items=int(np.round(train_size*batch_size))
    for i in range(n_batches):
        batch=new_data[i*batch_size:(i+1)*batch_size]
        train_batch= batch[:n_train_items]
        test_batch=batch[n_train_items:]
        yield (train_batch,test_batch,i+1)
    

def load_data(attr,mini_batch_size=50,cv=0,test=False):
    print ("loading data...")
    with open("processed.pkl","rb") as f:
        x = pickle.load(f)
    revs, W, W2, word_idx_map, vocab, mairesse = x[0], x[1], x[2], x[3], x[4], x[5]
    
    def data_idx2vec(data):
        print(data.flatten().shape)
        return W[np.array(data.flatten(),dtype="int32")].reshape((data.shape[0],data.shape[1],,W.shape[1]))

    print ("data loaded!")
    charged_words=[]
    emof=open("Emotion_Lexicon.csv","r")
    csvf=csv.reader(emof, delimiter=',',quotechar='"')
    first_line=True

    for line in csvf:
        if first_line:
            first_line=False
            continue
        if line[11]=="1":
            charged_words.append(line[0])

    emof.close()

    charged_words=set(charged_words)

    results = []
    #cv fold
    if cv <11:
        #datasets:[fortrainX, trainY, testX, testY, mTrain, mTest]
        datasets = make_idx_data_cv(revs, word_idx_map, mairesse, charged_words, cv, attr, max_l=149, max_s=312, filter_h=3)
        #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate
        #extra data (at random)
        if test==True:
            test_set_x = datasets[2].reshape((datasets[2].shape[0],-1))
            test_set_y = np.asarray(datasets[3],int).reshape((datasets[3].shape[0],-1))
            test_set_m = datasets[5].reshape((datasets[5].shape[0],-1))
            yield test_set_x,test_set_y,test_set_m
        else:
            mini_batches_generator=mini_batches(len(datasets[0]),batch_size=mini_batch_size)
            for t,v,_ in mini_batches_generator:
                    #divide train set into train/val sets
                    ##dataset shape :[trainX, trainY, testX, testY, mTrain, mTest]
                    #reshape (Minibatch,None)
                    train_set_x=datasets[0][t].reshape((t.shape[0],-1))
                    val_set_x=datasets[0][v].reshape((v.shape[0],-1))
                    train_set_y=datasets[1][t].reshape((t.shape[0],-1))
                    val_set_y=datasets[1][v].reshape((v.shape[0],-1))
                    #list of 84 feature per doc
                    train_set_m=datasets[4][t].reshape((t.shape[0],-1))
                    val_set_m=datasets[4][v].reshape((v.shape[0],-1))
                    #print('Mini-batch load : before transform idx to embed')
                    print(train_set_x.shape)
                    train_set_x=data_idx2vec(train_set_x)
                    val_set_x=data_idx2vec(val_set_x)
                    """
                    print(train_set_x.shape)
                    print(train_set_m.shape)
                    (45, 47736, 300)
                    (45, 84) 
                    """
                    yield train_set_x,train_set_y,val_set_x,val_set_y,train_set_m,val_set_m

                    # (45, 312, 153, 300)
                    #(batch,sentences_in_text,words_indexesin sentence)
                    #next(load_data(2))