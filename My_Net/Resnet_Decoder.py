import tensorflow as tf
from SIP_OPS import SIP_Conv2d
import tensorflow.contrib.slim as slim
from My_NN.My_Net.SIP_OPS import *
from tensorflow.python.platform import flags


class ResNet_Decoder(object):
    def __init__(self,final_endpoint,is_training):
        self.final_endpoint = final_endpoint
        self.is_training = is_training
        self.sip_conv2d = SIP_Conv2d()
        self.sip_bn = SIP_Batch_Norm()
        self.key_points = {}
        self.Flags = flags.FLAGS

    def get_variables(self):
        all_variables = tf.global_variables()
        self.train_variables = []
        self.train_variables_number = 0
        for v in all_variables:
            if (self.scope is not None and self.scope in v.name)\
                    or (self.scope is None and "ResNet_Decoder" in v.name):
                self.train_variables.append(v)
                tmp_num = 1
                for n in v.shape:
                    tmp_num *= n.value
                self.train_variables_number += tmp_num
        return self.train_variables

    def ops(self,input, scope = "ResNet_Decoder",reuse = False):
        sipconv2d = SIP_Conv2d()
        sipconv2d.set_default(is_training=self.is_training)
        self.scope = "ResNet_Decoder"
        with tf.variable_scope(scope, "ResNet_Decoder", reuse=reuse):
            setting = self.get_default_setting()
            setting.reverse()
            res_net = input
            num_of_output = conv_size = stride = 0
            res_net = self.bottom_block(res_net,512)#!!!!
            for i, s in enumerate(setting):
                i = len(setting) - 1 - i
                num_of_output ,conv_size, stride= s
                tmp_end_point_name = "D_Block_{0}_O_{1}_C_{2}_S_{3}".format(i, num_of_output, conv_size, stride)
                with tf.variable_scope(tmp_end_point_name):
                    res_net = self.res_block(res_net,num_of_output, conv_size, stride)
                    self.key_points[tmp_end_point_name] = res_net
                    if self.final_endpoint == tmp_end_point_name:
                        break
            res_net = self.top_block(res_net)
            self.get_variables()
            return res_net




    def res_block(self, input, num_of_output, conv_size, stride = 1, scope = "D_ResNet_Block"):
        with tf.variable_scope(scope,"D_ResNet_Block"):
            res_output = self.build_block(input, num_of_output, conv_size, stride)
            res_input = slim.conv2d_transpose(input,num_of_output,1,stride,"SAME",
                                              activation_fn = None,
                                              weights_initializer= tf.truncated_normal_initializer(stddev=self.Flags.Conv_W_init_stddev),
                                              weights_regularizer= slim.l2_regularizer(self.Flags.weight_decay),
                                              )
            res_input = self.sip_bn.ops(res_input,self.is_training)
            res_input = tf.nn.relu(res_input)
            res_net = res_output + res_input
            return res_net

    def build_block(self, input, num_of_output, conv_size, stride = 1, scope = "D_ResNet_Block"):
        with tf.variable_scope(scope, "D_ResNet_Block"):
            with tf.variable_scope("build_block_dconv2d_1"):
                res_net = slim.conv2d_transpose(input, num_of_output, conv_size, 1, "SAME",
                                                activation_fn=None,
                                                weights_initializer=tf.truncated_normal_initializer(
                                                    stddev=self.Flags.Conv_W_init_stddev),
                                                weights_regularizer=slim.l2_regularizer(self.Flags.weight_decay),
                                                )
                res_net = self.sip_bn.ops(res_net, self.is_training)
                res_net = tf.nn.relu(res_net)
            with tf.variable_scope(scope, "build_block_dconv2d_2"):
                res_net = slim.conv2d_transpose(res_net, num_of_output, conv_size, stride, "SAME",
                                                activation_fn=None,
                                                weights_initializer=tf.truncated_normal_initializer(
                                                    stddev=self.Flags.Conv_W_init_stddev),
                                                weights_regularizer=slim.l2_regularizer(self.Flags.weight_decay),
                                                )
                res_net = self.sip_bn.ops(res_net, self.is_training)
                res_net = tf.nn.relu(res_net)
            return res_net

    def get_default_setting(self):
        return [[64, 3, 1],
                [64, 3, 1],
                [64, 3, 1],
                [128, 3, 2],
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
                [512, 3, 1], ]

    def bottom_block(self,input,num_of_output):
        with tf.variable_scope("D_Bottom_Block"):
            with tf.variable_scope("bottom_block_dconv2d_1"):
                res_net = slim.conv2d_transpose(input, num_of_output, 1, 1, "SAME",
                                                activation_fn=None,
                                                weights_initializer=tf.truncated_normal_initializer(
                                                    stddev=self.Flags.Conv_W_init_stddev),
                                                weights_regularizer=slim.l2_regularizer(self.Flags.weight_decay),
                                                )
            with tf.variable_scope("bottom_block_dconv2d_2"):
                res_net = slim.conv2d_transpose(res_net, num_of_output, 3, 1, "SAME",
                                                activation_fn=None,
                                                weights_initializer=tf.truncated_normal_initializer(
                                                    stddev=self.Flags.Conv_W_init_stddev),
                                                weights_regularizer=slim.l2_regularizer(self.Flags.weight_decay),
                                                )
                res_net = self.sip_bn.ops(res_net, self.is_training)
                res_net = tf.nn.relu(res_net)
            with tf.variable_scope("bottom_block_dconv2d_3"):
                res_net = slim.conv2d_transpose(res_net, num_of_output, 1, 1, "SAME",
                                                activation_fn=None,
                                                weights_initializer=tf.truncated_normal_initializer(
                                                    stddev=self.Flags.Conv_W_init_stddev),
                                                weights_regularizer=slim.l2_regularizer(self.Flags.weight_decay),
                                                )
                res_net = self.sip_bn.ops(res_net, self.is_training)
                res_net = tf.nn.relu(res_net)
                return res_net

    def top_block(self, input):
        with tf.variable_scope("Top_Block"):
            with tf.variable_scope("top_block_conv2d_1"):
                res_net = slim.conv2d_transpose(input, 64, 1, 2, "SAME",
                                                activation_fn=None,
                                                weights_initializer=tf.truncated_normal_initializer(
                                                    stddev=self.Flags.Conv_W_init_stddev),
                                                weights_regularizer=slim.l2_regularizer(self.Flags.weight_decay),
                                                )
                res_net = self.sip_bn.ops(res_net, self.is_training)
                res_net = tf.nn.relu(res_net)
            with tf.variable_scope("top_block_conv2d_2"):
                res_net = slim.conv2d_transpose(res_net, 3, 7, 2, "SAME",
                                                activation_fn=None,
                                                weights_initializer=tf.truncated_normal_initializer(
                                                    stddev=self.Flags.Conv_W_init_stddev),
                                                weights_regularizer=slim.l2_regularizer(self.Flags.weight_decay),
                                                )
                return res_net
