from model import BigFiveCnnModel

filter_hs=[1,2,3]
img_w = 300
img_h=153#max sentence length
filter_w = img_w
feature_maps = [200,200,2]
filter_shapes = []
pool_sizes = []
for filter_h in filter_hs:
    filter_shapes.append((feature_maps, 1, filter_h, filter_w))
    pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
Model=BigFiveCnnModel(filter_shapes,pool_sizes,input_shape=(312, 153, 300),filter_hs=filter_hs)
print(Model.summary())