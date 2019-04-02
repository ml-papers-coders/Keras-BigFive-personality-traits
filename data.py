import numpy as np 
import csv
import pickle
import random

seed=7
np.random.seed(seed)

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



def load_data(attr,data_aug=False):
    print ("loading data...")
    with open("processed.pkl","rb") as f:
        x = pickle.load(f)
    revs, W, W2, word_idx_map, vocab, mairesse = x[0], x[1], x[2], x[3], x[4], x[5]
    
    def augment(element_rev):
        nb=random.randint(0,int(len(element_rev["text"])//2))
        for i in range(nb):
            element_rev["text"]=element_rev["text"].pop(random.randrange(len(element_rev["text"])))

    if data_aug==True:
        revs2=list(map(augment,revs))
        revs=revs+revs2
        mairesse=mairesse+mairesse
        print('Data Augmentation...')
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


def data_idx(data_size,batch_size):
    
    if data_size % batch_size > 0 :
        extra_data_num = batch_size - data_size % batch_size
        
        rand_perm = np.random.permutation(range(data_size))
        extra_data=rand_perm[:extra_data_num]
        
        # add random replication
        new_data=np.append(np.arange(data_size),extra_data)
    else:
        new_data=np.random.permutation(range(data_size))
    return new_data    

def data_gen(attr,data_idx,datasets,W,batch_size,test=False):
    #n_batches = int(data_idx.shape[0]//batch_size)
    #_S,_W=reshape
    while True:
        # Shuffling the new data
        rand_perm = np.random.permutation(range(data_idx.shape[0]))
        data_idx=data_idx[rand_perm]
        batch_idx=data_idx[:batch_size]
        

        train_set_x=datasets[0][batch_idx]
        #list of 84 feature per doc
        train_set_m=datasets[2][batch_idx].reshape((-1,84))
        #if test==False:
        train_set_y=datasets[1][batch_idx].reshape((-1,1)) # -1 W,E,1
        yield [train_set_x,train_set_m],train_set_y
        #else :
        #    yield [train_set_x,train_set_m]

def tfgenerator(datasets):
    for i in len(datasets[0]):
        train_set_x=np.asarray(datasets[0][i],np.int32)
        train_set_y=np.asarray(datasets[1][i].reshape((1,)),np.int8)
        train_set_m=np.asarray(datasets[2][i].reshape((84,)),np.float32)
        yield (train_set_x,train_set_m)