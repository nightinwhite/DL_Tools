#coding:utf-8
import tensorflow as tf
from tensorflow.python.platform import flags
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib import layers
from tensorflow.python.framework import ops
class SIP_Conv2d():
    def __init__(self):
        self.num_outputs = None
        self.kernel_size = None
        self.stride = None
        self.padding = None
        self.scope = None
        self.istraining = None

        Flags = flags.FLAGS
        self.default_config = {
            "weights_regularizer" : regularizers.l2_regularizer(Flags.weight_decay),
            "weights_initializer" : tf.truncated_normal_initializer(stddev=Flags.Conv_W_init_stddev),
            "batch_norm" : SIP_Batch_Norm(),
            "activation_fn" : tf.nn.relu
        }#整体网络的会变的量

    def set_default(self,num_outputs = None,kernel_size = None, stride = None,padding = None,scope = None,activation_fn = None,batch_norm = None, is_training = None):
        #单个节点会变的量
        if num_outputs is not None:
            self.default_config["num_outputs"] = num_outputs
        if kernel_size is not None:
            self.default_config["kernel_size"] = kernel_size
        if stride is not None:
            self.default_config["stride"] = stride
        if padding is not None:
            self.default_config["padding"] = padding
        if scope is not None:
            self.default_config["scope"] = scope
        if activation_fn is not None:
            self.default_config["activation_fn"] = activation_fn
        if is_training is not None:
            self.default_config["is_training"] = is_training
        if batch_norm is not None:
            self.default_config["batch_norm"] = batch_norm

    def ops(self,input,num_outputs = None,kernel_size = None, stride = None,padding = None,scope = None,is_training = None,activation_fn = None):
        if num_outputs is None:
            num_outputs = self.default_config["num_outputs"]
        if kernel_size is None:
            kernel_size = self.default_config["kernel_size"]
        if stride is None:
            stride = self.default_config["stride"]
        if padding is None:
            padding = self.default_config["padding"]
        if scope is None:
            scope = self.default_config["scope"]
        if is_training is None:
            is_training = self.default_config["is_training"]
        if activation_fn is None:
            activation_fn = self.default_config["activation_fn"]

        tmp_batch_norm = self.default_config["batch_norm"]
        with tf.variable_scope(scope, "SIP_Conv2d"):
            net = layers.conv2d(input, num_outputs, kernel_size, stride=stride,
                                padding=padding,
                                weights_regularizer=self.default_config["weights_regularizer"],
                                weights_initializer=self.default_config["weights_initializer"],
                                )
            if tmp_batch_norm is not None:
                net = tmp_batch_norm.ops(net,is_training = is_training)
            if activation_fn is not None:
                net = activation_fn(net)
            return net



class SIP_Batch_Norm():
    def __init__(self):
        self.decay = None
        self.epsilon = None
        self.updates_collections = None
        self.is_training = None
        self.default_config = {
            "decay": 0.9997,
            "epsilon": 0.01,
            "updates_collections": ops.GraphKeys.UPDATE_OPS,
        }

    def set_default(self,decay = None, epsilon = None, updates_collections = None,is_training = None):
        if is_training is not None:
            self.default_config["is_training"] = is_training
        if decay is not None:
            self.default_config["decay"] = decay
        if epsilon is not None:
            self.default_config["epsilon"] = epsilon
        if updates_collections is not None:
            self.default_config["updates_collections"] = updates_collections

    def ops(self,input,is_training = None):
        if is_training is None:
            is_training = self.default_config["is_training"]
        with tf.variable_scope("batch_norm"):
            return slim.batch_norm(inputs=input,
                                   decay=self.default_config["decay"],
                                   epsilon=self.default_config["epsilon"],
                                   updates_collections=self.default_config["updates_collections"],
                                   is_training=is_training)
