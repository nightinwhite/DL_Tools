import tensorflow as tf
from My_Net.Inception_v3 import Inception_v3

import My_NN.My_Net.Test_for_NN.common_flags

My_NN.My_Net.Test_for_NN.common_flags.define()

img = tf.placeholder(tf.float32,[None,299,299,3],"img")
inception_nn = Inception_v3()
net = inception_nn.ops(img)
inception_nn.get_variables()
print net.get_shape()
print inception_nn.train_variables_number
