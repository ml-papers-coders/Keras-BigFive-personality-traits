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

def w2idx(revs, word_idx_map, mairesse, charged_words, per_attr=0, max_l=51, max_s=200, filter_h=5):
    """
    Transforms sentences into a 2-d matrix. of word indx
    """
    trainX, trainY, mTrain = [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map,
        charged_words,
        max_l, max_s, filter_h)
        
        trainX.append(sent)
        trainY.append(rev['y'+str(per_attr)])
        mTrain.append(mairesse[rev["user"]])
    trainX = np.array(trainX,dtype=int)
    #mTrain is the mairesse features
    mTrain = np.array(mTrain, dtype=float)
    trainY=np.array(trainY,dtype=int)
    return [trainX, trainY , mTrain]



def load_data(attr):
    print ("loading data...")
    with open("processed.pkl","rb") as f:
        x = pickle.load(f)
    revs, W, W2, word_idx_map, vocab, mairesse = x[0], x[1], x[2], x[3], x[4], x[5]

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

    return revs, W, W2, word_idx_map, vocab, mairesse ,charged_words

def data_idx2vec(data,W):
    print(data.flatten().shape)
    return np.asarray(W[np.array(data.flatten(),dtype="int32")]).reshape((data.shape[0],data.shape[1],data.shape[2],W.shape[1]))


def data_idx(attr,data_size,batch_size,seed=0):
    np.random.seed(seed)
    if data_size % batch_size > 0 :
        extra_data_num = batch_size - data_size % batch_size
        
        rand_perm = np.random.permutation(range(data_size))
        extra_data=rand_perm[:extra_data_num]
        
        # add random replication
        new_data=np.append(np.arange(data_size),extra_data)
    else:
        new_data=np.random.permutation(range(data_size))
    return new_data    

def data_gen(attr,data_idx,datasets,W,batch_size,reshape,seed=0):
    n_batches = int(data_idx.shape[0]/batch_size)
    _S,_W=reshape
    for i in range(n_batches):
        # Shuffling the new data
        rand_perm = np.random.permutation(range(len(data_idx)))
        data_idx=data_idx[rand_perm]
        batch_idx=data_idx[:batch_size]
        
        train_set_x=datasets[0][batch_idx]
        train_set_y=datasets[1][batch_idx].reshape((-1,1)) # -1 W,E,1
        #list of 84 feature per doc
        train_set_m=datasets[2][batch_idx].reshape((-1,84))
        #print('Mini-batch load : before transform idx to embed')
        #print(train_set_x.shape)
        #print(train_set_x.dtype) int64
        #train_set_x=data_idx2vec(train_set_x,W)
        _E=W.shape[1]
        #train_set_x=train_set_x.reshape((-1,_S,_W))
        print(train_set_x.shape)
        print(train_set_y.shape)
    
        """
        print(train_set_x.shape)
        print(train_set_m.shape)
        (45, 312, 153, 300)
        (45, 84)
        """
        print("batch:"+str(i))
        yield [train_set_x,train_set_m],train_set_y

