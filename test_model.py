from model import SentenceLevel,DocumentLevel
from tensorflow.keras.models import Model


"""
img_w=E
img_h=Sentence length
"""
filter_hs=[1,2,3]
img_w = 300
img_h=153#max sentence length
filter_w = img_w
feature_maps = [200,200,2]
filter_shapes = []
pool_sizes = []
for filter_h in filter_hs:
    filter_shapes.append((filter_h,filter_w))
    pool_sizes.append((img_h-filter_h+1,img_w-filter_w+1))
#input_shape=(312,153, 300) : nb_sents X nb_words X E
#Batch : (batch=nb_docs)*nb sentences
inputLayer,concatlayer=SentenceLevel(filter_shapes,pool_sizes,input_shape=(153,300,1),filter_hs=filter_hs)

docs_size=312
outputLayer=DocumentLevel(concatlayer,docs_size)
print(Model(inputs=inputLayer,outputs=outputLayer).summary())

"""
from tensorflow.keras.utils import plot_model
plot_model(Model(inputs=inputLayer,outputs=outputLayer), to_file='model.png')
"""
