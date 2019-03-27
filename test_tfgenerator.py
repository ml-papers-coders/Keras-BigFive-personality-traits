from data import data_idx,load_data,w2idx,data_gen,tfgenerator
import tensorflow as tf



def test(attr):
    revs, W, W2, word_idx_map, vocab, mairesse ,charged_words=load_data(attr)
    datasets = w2idx(revs, word_idx_map, mairesse, charged_words, attr, max_l=149, max_s=312, filter_h=3)
    dataset=tf.data.Dataset.from_generator(generator=tfgenerator,output_types=((tf.int32,tf.float32),tf.int8),args=attr)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        try: 
            # Keep running next_batch till the Dataset is exhausted
            while True:
                print(sess.run(next_element))
                
        except tf.errors.OutOfRangeError:
            pass





test(attr=2)