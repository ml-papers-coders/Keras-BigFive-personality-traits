from model import BigFiveCnnModel
from tensorflow.keras import backend as K
import tensorflow as tf
from data import data_idx,load_data,w2idx,data_gen
import numpy as np
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard

LOG_DIR = os.path.join(ROOT, 'log1')

def tb_config():
    import os
    import requests
    import shutil
    import subprocess

    __all__ = [
    'install_ngrok', 
    'launch_tensorboard',
    ]

    def __shell__(cmd, split=True):
    # get_ipython().system_raw(cmd)
    result = get_ipython().getoutput(cmd, split=split)
    if result and not split:
        result = result.strip('\n')
    return result  


    # tested OK
    def install_ngrok(bin_dir="/tmp"):
    """ download and install ngrok on local vm instance
    Args:
        bin_dir: full path for the target directory for the `ngrok` binary
    """
    TARGET_DIR = bin_dir
    CWD = os.getcwd()
    is_grok_avail = os.path.isfile(os.path.join(TARGET_DIR,'ngrok'))
    if is_grok_avail:
        print("ngrok installed")
    else:
        import platform
        plat = platform.platform() # 'Linux-4.4.64+-x86_64-with-Ubuntu-17.10-artful'
        if 'x86_64' in plat:
        
        os.chdir('/tmp')
        print("calling wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip ..." )
        get_ipython().system_raw( "wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip" )
        print("calling unzip ngrok-stable-linux-amd64.zip ...")
        get_ipython().system_raw( "unzip ngrok-stable-linux-amd64.zip" )
        os.rename("ngrok", "{}/ngrok".format(TARGET_DIR))
        os.remove("ngrok-stable-linux-amd64.zip")
        is_grok_avail = os.path.isfile(os.path.join(TARGET_DIR,'ngrok'))
        os.chdir(TARGET_DIR)
        if is_grok_avail:
            print("ngrok installed. path={}".format(os.path.join(TARGET_DIR,'ngrok')))
        else:
            # ValueError: ERROR: ngrok not found, path=
            raise ValueError( "ERROR: ngrok not found, path=".format(TARGET_DIR) )
        else:
        raise NotImplementedError( "ERROR, ngrok install not configured for this platform, platform={}".format(plat))
        os.chdir(CWD)
        return
        
    # tested OK
    def launch_tensorboard(bin_dir="/tmp", log_dir="/tmp", retval=False):
    """returns a public tensorboard url based on the ngrok package
    checks if `ngrok` is available, and installs, if necessary, to `bin_dir`
    launches tensorboard, if necessary
    see: https://stackoverflow.com/questions/47818822/can-i-use-tensorboard-with-google-colab
    Args:
        bin_dir: full path for the target directory for the `ngrok` binary
        log_dir: full path for the tensorflow `log_dir`
    Return:
        public url for tensorboard if retval==True
        NOTE: the method will print a link to stdout (cell output) for the tensorflow URL. 
        But the link printed from the return value has an extra "%27" in the URL which causes an error
    """
    install_ngrok(bin_dir)
        
    if not tf.gfile.Exists(log_dir):  tf.gfile.MakeDirs(log_dir)
    
    # check status of tensorboard and ngrok
    ps = __shell__("ps -ax")
    is_tensorboard_running = len([f for f in ps if "tensorboard" in f ]) > 0
    is_ngrok_running = len([f for f in ps if "ngrok" in f ]) > 0
    print("status: tensorboard={}, ngrok={}".format(is_tensorboard_running, is_ngrok_running))

    if not is_tensorboard_running:
        get_ipython().system_raw(
            'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
            .format(log_dir)
        )
        is_tensorboard_running = True
        
    if not is_ngrok_running:  
        #    grok should be installed in /tmp/ngrok
        get_ipython().system_raw('{}/ngrok http 6006 &'.format(bin_dir))
        is_ngrok_running = True

    # get tensorboard url
    # BUG: getting connection refused for HTTPConnectionPool(host='localhost', port=4040)
    #     on first run, retry works
    import time
    time.sleep(3)
    retval = requests.get('http://localhost:4040/api/tunnels')
    tensorboard_url = retval.json()['tunnels'][0]['public_url'].strip()
    print("tensorboard url=", tensorboard_url)
    if retval:
        return tensorboard_url
    # set paths
    ROOT = %pwd

    # will install `ngrok`, if necessary
    # will create `log_dir` if path does not exist
    launch_tensorboard( bin_dir=ROOT, log_dir=LOG_DIR )

def nll1(y_true, y_pred):
    """ Negative log likelihood. """

    # keras.losses.binary_crossentropy give the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

def nll2(y_true, y_pred):
    """ Negative log likelihood. """

    likelihood = tf.distributions.Bernoulli(probs=y_pred)

    return - K.sum(likelihood.log_prob(y_true), axis=-1)

def init(attr=2,train_size=0.9,batcTensorBoardh_size=25):
    #data generator
    #load first mini-batch
    """
    train_set_x (45, 312, 153, 300)
    """
    
    revs, W, W2, word_idx_map, vocab, mairesse ,charged_words=load_data(attr)
    datasets = w2idx(revs, word_idx_map, mairesse, charged_words, attr, max_l=149, max_s=312, filter_h=3)
    _D=len(datasets[0])
    _S=len(datasets[0][0])
    _W=len(datasets[0][0][0])
    _E=W.shape[1]
    dataset_idx=data_idx(attr,len(datasets[0]),batch_size)

    #split train val
    n_train_items=int(np.round(train_size*_D))
    train_idx=dataset_idx[:n_train_items]
    test_idx=dataset_idx[n_train_items:]
    train_generator=data_gen(attr,train_idx,datasets,W,reshape=(_S,_W),batch_size=25)
    test_generator=data_gen(attr,test_idx,datasets,W,reshape=(_S,_W),batch_size=25)
    input_shape=(_S*_W,_E,1)
    docs_size=_S
    hidden_units=[200,200,2]
    filter_hs=[1,2,3]
    filter_shapes = []
    pool_sizes = []
    reshape=(_S,_W)
    for filter_h in filter_hs:
        filter_shapes.append((filter_h, _E))
        pool_sizes.append((_S*(_W-filter_h+1),1))
    model=BigFiveCnnModel(W,filter_shapes,pool_sizes,reshape,filter_hs=filter_hs,hidden_units=hidden_units,docs_size=docs_size)
    model.summary()
    model.compile(loss=nll1,optimizer="adadelta")
    steps=train_idx.shape[0]//batch_size
    v_steps=test_idx.shape[0]//batch_size
    return model,train_generator,test_generator,steps,v_steps

    
def train(batch_size,attr=2):

    with tf.device('/cpu:0'):
        model,train_generator,test_generator,steps,vsteps=init(attr,batch_size=batch_size)
        #take a selfie of the model :D
        plot_model(model, to_file='selfie.png')
    with tf.device('/gpu:0'):
        print('=================== Training ===================')
        model.fit_generator(
        generator=train_generator,
        epochs=1,
        validation_data=test_generator,
        steps_per_epoch=steps//10
        ,validation_steps=vsteps//10
        ,callbacks=[TensorBoard(
            log_dir=LOG_DIR, histogram_freq=0
            , write_graph=True
            , write_images=True)]
        )
        model.save("model.h5")



    
        
    
train(batch_size=5)

