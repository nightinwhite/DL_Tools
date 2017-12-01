import tensorflow as tf
from SIP_OPS import SIP_Conv2d
import tensorflow.contrib.slim as slim
from My_NN.My_Net.SIP_OPS import *

class ResNet_Encoder(object):
    def __init__(self,final_endpoint,is_training):
        self.final_endpoint = final_endpoint
        self.is_training = is_training
        self.sip_conv2d = SIP_Conv2d()
        self.key_points = {}

    def get_variables(self):
        all_variables = tf.global_variables()
        self.train_variables = []
        self.train_variables_number = 0
        for v in all_variables:
            if (self.scope is not None and self.scope in v.name)\
                    or (self.scope is None and "ResNet_Encoder" in v.name):
                self.train_variables.append(v)
                tmp_num = 1
                for n in v.shape:
                    tmp_num *= n.value
                self.train_variables_number += tmp_num
        return self.train_variables

    def ops(self,input, scope = "ResNet_Encoder",reuse = False):
        sipconv2d = SIP_Conv2d()
        sipconv2d.set_default(is_training=self.is_training)
        self.scope = "ResNet_Encoder"
        with tf.variable_scope(scope, "ResNet_Encoder", reuse=reuse):
            setting = self.get_default_setting()
            res_net = input
            num_of_output = conv_size = stride = 0
            res_net = self.top_block(res_net)
            for i, s in enumerate(setting):
                num_of_output ,conv_size, stride= s
                tmp_end_point_name = "Block{0}_O_{1}_C_{2}_S_{3}".format(i, num_of_output, conv_size, stride)
                with tf.variable_scope(tmp_end_point_name):
                    res_net = self.res_block(res_net,num_of_output, conv_size, stride)
                    self.key_points[tmp_end_point_name] = res_net
                    if self.final_endpoint == tmp_end_point_name:
                        break
            res_net = self.bottom_block(res_net,num_of_output)
            self.get_variables()
            return res_net

    def res_block(self, input, num_of_output, conv_size, stride = 1, scope = "ResNet_Block"):
        with tf.variable_scope(scope,"ResNet_Block"):
            res_output = self.build_block(input, num_of_output,conv_size, stride)
            res_input = self.sip_conv2d.ops(input,num_of_output,1,stride,
                                            padding="SAME", scope = "res_input_conv2d",is_training=self.is_training)
            res_net = res_output + res_input
            return res_net

    def build_block(self, input, num_of_output, conv_size, stride = 1, scope = "Build_Block"):
        with tf.variable_scope(scope, "ResNet_Block"):
            res_net = self.sip_conv2d.ops(input, num_of_output, conv_size, stride,
                                          padding="SAME", scope="build_block_conv2d_1", is_training=self.is_training)
            res_net = self.sip_conv2d.ops(res_net, num_of_output, conv_size, 1,
                                          padding="SAME", scope="build_block_conv2d_2", is_training=self.is_training)
            return res_net

    def get_default_setting(self):
        return [[64, 3, 1],
                [64, 3, 1],
                [64, 3, 1],
                [128,3, 2],
                [128, 3, 1],
                [128, 3, 1],
                [128, 3, 1],
                [256, 3, 2],
                [256, 3, 1],
                [256, 3, 1],
                [256, 3, 1],
                [256, 3, 1],
                [256, 3, 1],
                [512, 3, 2],
                [512, 3, 1],
                [512, 3, 1],]

    def bottom_block(self,input,num_of_output):
        with tf.variable_scope("Bottom_Block"):
            res_input = self.sip_conv2d.ops(input, num_of_output, 1, 1,
                                            padding="SAME", scope="bottom_block_conv2d_1", is_training=self.is_training)
            res_input = self.sip_conv2d.ops(res_input, num_of_output, 3, 1,
                                            padding="SAME", scope="bottom_block_conv2d_2", is_training=self.is_training)
            self.sip_conv2d.default_config["batch_norm"] = None
            self.sip_conv2d.default_config["weights_regularizer"] = None
            self.sip_conv2d.default_config["activation_fn"] = None
            res_input = self.sip_conv2d.ops(res_input, num_of_output, 1, 1,
                                            padding="SAME", scope="bottom_block_conv2d_3", is_training=self.is_training)
            return res_input

    def top_block(self,input):
        with tf.variable_scope("Top_Block"):
            res_input = self.sip_conv2d.ops(input, 64, 7, 2,
                                            padding="SAME", scope="top_block_conv2d_1", is_training=self.is_training)
            res_input = self.sip_conv2d.ops(res_input, 64, 1, 2,
                                            padding="SAME", scope="top_block_conv2d_2", is_training=self.is_training)
            return res_input
