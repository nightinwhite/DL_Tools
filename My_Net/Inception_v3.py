#coding:utf-8
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from SIP_OPS import SIP_Conv2d
class Inception_v3(object):
    def __init__(self,final_endpoint,min_depth,depth_multiplier,is_training):
        self.final_endpoint = final_endpoint
        self.min_depth = min_depth
        self.depth_multiplier = depth_multiplier
        self.is_training = is_training
        self.key_point = {}

    def get_variables(self):
        all_variables = tf.global_variables()
        self.train_variables = []
        self.train_variables_number = 0
        for v in all_variables:
            if (self.scope is not None and self.scope in v.name)\
                    or (self.scope is None and "Inception_v3" in v.name):
                self.train_variables.append(v)
                tmp_num = 1
                for n in v.shape:
                    tmp_num *= n.value
                self.train_variables_number += tmp_num
        return self.train_variables


    def ops(self,input,scope=None,reuse = False):
        self.scope = scope
        sipconv2d = SIP_Conv2d()
        sipconv2d.set_default(is_training=self.is_training)
        if self.depth_multiplier <= 0:
            raise ValueError('depth_multiplier is not greater than zero.')
        depth = lambda d: max(int(d * self.depth_multiplier), self.min_depth)
        with tf.variable_scope(scope,"Inception_v3",reuse=reuse):
            sipconv2d.set_default(stride=1, padding="VALID")
            #---------------------------------------
            end_point = "Conv2d_p1_3x3"
            # ---- ops
            net = sipconv2d.ops(input,depth(32),[3, 3],stride=2,scope=end_point)
            # ----
            self.key_point[end_point] = net
            if end_point == self.final_endpoint:
                self.get_variables()
                return net
            # ---------------------------------------
            end_point = "Conv2d_p2_3x3"
            # ---- ops
            net = sipconv2d.ops(net,depth(32),[3, 3],scope=end_point)
            # ----
            self.key_point[end_point] = net
            if end_point == self.final_endpoint:
                self.get_variables()
                return net
            # ---------------------------------------
            end_point = "Conv2d_p3_3x3"
            # ---- ops
            net = sipconv2d.ops(net,depth(64),[3, 3],padding="SAME",scope=end_point)
            # ----
            self.key_point[end_point] = net
            if end_point == self.final_endpoint:
                self.get_variables()
                return net
            # ---------------------------------------
            end_point = "MaxPool_p4_3x3"
            # ---- ops
            net = layers_lib.max_pool2d(net,[3,3],stride=2,scope=end_point)
            # ----
            self.key_point[end_point] = net
            if end_point == self.final_endpoint:
                self.get_variables()
                return net
            # ---------------------------------------
            end_point = "Conv2d_p5_1x1"
            # ---- ops
            net = sipconv2d.ops(net,depth(80),[1, 1],scope=end_point)
            # ----
            self.key_point[end_point] = net
            if end_point == self.final_endpoint:
                self.get_variables()
                return net
            # ---------------------------------------
            end_point = "Conv2d_p6_3x3"
            # ---- ops
            net = sipconv2d.ops(net,depth(192),[3, 3],scope=end_point)
            # ----
            self.key_point[end_point] = net
            if end_point == self.final_endpoint:
                self.get_variables()
                return net
            # ---------------------------------------
            end_point = "MaxPool_p7_3x3"
            # ---- ops
            net = layers_lib.max_pool2d(net,[3,3],stride=2,scope=end_point)
            # ----
            self.key_point[end_point] = net
            if end_point == self.final_endpoint:
                self.get_variables()
                return net
            # ---------------------------------------
            end_point = "Mixed_p8_a"
            # ---- ops
            sipconv2d.set_default(padding="SAME",stride=1)
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = sipconv2d.ops(net,depth(64),[1,1],scope="a_conv2d")

                with tf.variable_scope('Branch_1'):
                    branch_1 = sipconv2d.ops(net,depth(48),[1,1],scope="b_conv2d_1")
                    branch_1 = sipconv2d.ops(branch_1, depth(64), [5, 5], scope='b_conv2d_2')

                with tf.variable_scope('Branch_2'):
                    branch_2 = sipconv2d.ops(net, depth(64), [1, 1], scope="c_conv2d_1")
                    branch_2 = sipconv2d.ops(branch_2, depth(96), [3, 3], scope="c_conv2d_2")
                    branch_2 = sipconv2d.ops(branch_2, depth(96), [3, 3], scope="c_conv2d_3")

                with tf.variable_scope('Branch_3'):
                    branch_3 = layers_lib.avg_pool2d(net, [3, 3], stride=1,padding="SAME",scope='d_AvgPool')
                    branch_3 = sipconv2d.ops(branch_3, depth(32), [1, 1], scope='d_conv2d')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            # ----
            self.key_point[end_point] = net
            if end_point == self.final_endpoint:
                self.get_variables()
                return net
            # ---------------------------------------
            end_point = "Mixed_p8_b"
            # ---- ops
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = sipconv2d.ops(net,depth(64),[1,1],scope="a_conv2d")

                with tf.variable_scope('Branch_1'):
                    branch_1 = sipconv2d.ops(net,depth(48),[1,1],scope="b_conv2d_1")
                    branch_1 = sipconv2d.ops(branch_1, depth(64), [5, 5], scope='b_conv2d_2')

                with tf.variable_scope('Branch_2'):
                    branch_2 = sipconv2d.ops(net, depth(64), [1, 1], scope="c_conv2d_1")
                    branch_2 = sipconv2d.ops(branch_2, depth(96), [3, 3], scope="c_conv2d_2")
                    branch_2 = sipconv2d.ops(branch_2, depth(96), [3, 3], scope="c_conv2d_3")

                with tf.variable_scope('Branch_3'):
                    branch_3 = layers_lib.avg_pool2d(net, [3, 3], stride=1, padding="SAME",scope='d_AvgPool')
                    branch_3 = sipconv2d.ops(branch_3, depth(64), [1, 1], scope='d_conv2d')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            # ----
            self.key_point[end_point] = net
            if end_point == self.final_endpoint:
                self.get_variables()
                return net
            # ---------------------------------------
            end_point = "Mixed_p8_c"
            # ---- ops
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = sipconv2d.ops(net,depth(64),[1,1],scope="a_conv2d")

                with tf.variable_scope('Branch_1'):
                    branch_1 = sipconv2d.ops(net,depth(48),[1,1],scope="b_conv2d_1")
                    branch_1 = sipconv2d.ops(branch_1, depth(64), [5, 5], scope='b_conv2d_2')

                with tf.variable_scope('Branch_2'):
                    branch_2 = sipconv2d.ops(net, depth(64), [1, 1], scope="c_conv2d_1")
                    branch_2 = sipconv2d.ops(branch_2, depth(96), [3, 3], scope="c_conv2d_2")
                    branch_2 = sipconv2d.ops(branch_2, depth(96), [3, 3], scope="c_conv2d_3")

                with tf.variable_scope('Branch_3'):
                    branch_3 = layers_lib.avg_pool2d(net, [3, 3], stride=1, padding="SAME",scope='d_AvgPool')
                    branch_3 = sipconv2d.ops(branch_3, depth(64), [1, 1], scope='d_conv2d')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            # ----
            self.key_point[end_point] = net
            if end_point == self.final_endpoint:
                self.get_variables()
                return net
            # ---------------------------------------
            end_point = "Mixed_p9_a"
            # ---- ops
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = sipconv2d.ops(net,depth(384),[3,3],stride=2,padding="VALID",scope="a_conv2d")

                with tf.variable_scope('Branch_1'):
                    branch_1 = sipconv2d.ops(net,depth(64),[1,1],scope="b_conv2d_1")
                    branch_1 = sipconv2d.ops(branch_1, depth(96), [3, 3], scope='b_conv2d_2')
                    branch_1 = sipconv2d.ops(branch_1, depth(96), [3, 3], stride=2,padding="VALID",scope='b_conv2d_3')

                with tf.variable_scope('Branch_2'):
                    branch_2 = layers_lib.max_pool2d(net,[3,3],stride=2,padding="VALID",scope="c_Max_Pool")
                net = tf.concat([branch_0, branch_1, branch_2], 3)
            # ----
            self.key_point[end_point] = net
            if end_point == self.final_endpoint:
                self.get_variables()
                return net
            # ---------------------------------------
            end_point = "Mixed_p9_b"
            # ---- ops
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = sipconv2d.ops(net, depth(192), [1, 1], scope="a_conv2d")

                with tf.variable_scope('Branch_1'):
                    branch_1 = sipconv2d.ops(net, depth(128), [1, 1], scope="b_conv2d_1")
                    branch_1 = sipconv2d.ops(branch_1, depth(128), [1, 7], scope='b_conv2d_2')
                    branch_1 = sipconv2d.ops(branch_1, depth(192), [7, 1], scope='b_conv2d_3')

                with tf.variable_scope('Branch_2'):
                    branch_2 = sipconv2d.ops(net, depth(128), [1, 1], scope="c_conv2d_1")
                    branch_2 = sipconv2d.ops(branch_2, depth(128), [7, 1], scope="c_conv2d_2")
                    branch_2 = sipconv2d.ops(branch_2, depth(128), [1, 7], scope="c_conv2d_3")
                    branch_2 = sipconv2d.ops(branch_2, depth(128), [7, 1], scope="c_conv2d_4")
                    branch_2 = sipconv2d.ops(branch_2, depth(192), [1, 7], scope="c_conv2d_5")

                with tf.variable_scope('Branch_3'):
                    branch_3 = layers_lib.avg_pool2d(net, [3, 3], stride=1, padding="SAME",scope='d_AvgPool')
                    branch_3 = sipconv2d.ops(branch_3, depth(192), [1, 1], scope='d_conv2d')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            # ----
            self.key_point[end_point] = net
            if end_point == self.final_endpoint:
                self.get_variables()
                return net
            # ---------------------------------------
            end_point = "Mixed_p9_c"
            # ---- ops
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = sipconv2d.ops(net, depth(192), [1, 1], scope="a_conv2d")

                with tf.variable_scope('Branch_1'):
                    branch_1 = sipconv2d.ops(net, depth(160), [1, 1], scope="b_conv2d_1")
                    branch_1 = sipconv2d.ops(branch_1, depth(160), [1, 7], scope='b_conv2d_2')
                    branch_1 = sipconv2d.ops(branch_1, depth(192), [7, 1], scope='b_conv2d_3')

                with tf.variable_scope('Branch_2'):
                    branch_2 = sipconv2d.ops(net, depth(160), [1, 1], scope="c_conv2d_1")
                    branch_2 = sipconv2d.ops(branch_2, depth(160), [7, 1], scope="c_conv2d_2")
                    branch_2 = sipconv2d.ops(branch_2, depth(160), [1, 7], scope="c_conv2d_3")
                    branch_2 = sipconv2d.ops(branch_2, depth(160), [7, 1], scope="c_conv2d_4")
                    branch_2 = sipconv2d.ops(branch_2, depth(192), [1, 7], scope="c_conv2d_5")

                with tf.variable_scope('Branch_3'):
                    branch_3 = layers_lib.avg_pool2d(net, [3, 3], stride=1, padding="SAME",scope='d_AvgPool')
                    branch_3 = sipconv2d.ops(branch_3, depth(192), [1, 1], scope='d_conv2d')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            # ----
            self.key_point[end_point] = net
            if end_point == self.final_endpoint:
                self.get_variables()
                return net
            # ---------------------------------------
            end_point = "Mixed_p9_d"
            # ---- ops
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = sipconv2d.ops(net, depth(192), [1, 1], scope="a_conv2d")

                with tf.variable_scope('Branch_1'):
                    branch_1 = sipconv2d.ops(net, depth(160), [1, 1], scope="b_conv2d_1")
                    branch_1 = sipconv2d.ops(branch_1, depth(160), [1, 7], scope='b_conv2d_2')
                    branch_1 = sipconv2d.ops(branch_1, depth(192), [7, 1], scope='b_conv2d_3')

                with tf.variable_scope('Branch_2'):
                    branch_2 = sipconv2d.ops(net, depth(160), [1, 1], scope="c_conv2d_1")
                    branch_2 = sipconv2d.ops(branch_2, depth(160), [7, 1], scope="c_conv2d_2")
                    branch_2 = sipconv2d.ops(branch_2, depth(160), [1, 7], scope="c_conv2d_3")
                    branch_2 = sipconv2d.ops(branch_2, depth(160), [7, 1], scope="c_conv2d_4")
                    branch_2 = sipconv2d.ops(branch_2, depth(192), [1, 7], scope="c_conv2d_5")

                with tf.variable_scope('Branch_3'):
                    branch_3 = layers_lib.avg_pool2d(net, [3, 3], stride=1, padding="SAME",scope='d_AvgPool')
                    branch_3 = sipconv2d.ops(branch_3, depth(192), [1, 1], scope='d_conv2d')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            # ----
            self.key_point[end_point] = net
            if end_point == self.final_endpoint:
                self.get_variables()
                return net
            # ---------------------------------------
            end_point = "Mixed_p9_e"
            # ---- ops
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = sipconv2d.ops(net, depth(192), [1, 1], scope="a_conv2d")

                with tf.variable_scope('Branch_1'):
                    branch_1 = sipconv2d.ops(net, depth(192), [1, 1], scope="b_conv2d_1")
                    branch_1 = sipconv2d.ops(branch_1, depth(192), [1, 7], scope='b_conv2d_2')
                    branch_1 = sipconv2d.ops(branch_1, depth(192), [7, 1], scope='b_conv2d_3')

                with tf.variable_scope('Branch_2'):
                    branch_2 = sipconv2d.ops(net, depth(192), [1, 1], scope="c_conv2d_1")
                    branch_2 = sipconv2d.ops(branch_2, depth(192), [7, 1], scope="c_conv2d_2")
                    branch_2 = sipconv2d.ops(branch_2, depth(192), [1, 7], scope="c_conv2d_3")
                    branch_2 = sipconv2d.ops(branch_2, depth(192), [7, 1], scope="c_conv2d_4")
                    branch_2 = sipconv2d.ops(branch_2, depth(192), [1, 7], scope="c_conv2d_5")

                with tf.variable_scope('Branch_3'):
                    branch_3 = layers_lib.avg_pool2d(net, [3, 3], stride=1, padding="SAME",scope='d_AvgPool')
                    branch_3 = sipconv2d.ops(branch_3, depth(192), [1, 1], scope='d_conv2d')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            # ----
            self.key_point[end_point] = net
            if end_point == self.final_endpoint:
                self.get_variables()
                return net
            # ---------------------------------------
            end_point = "Mixed_p10_a"
            # ---- ops
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = sipconv2d.ops(net,depth(192),[1,1],scope="a_conv2d_1")
                    branch_0 = sipconv2d.ops(branch_0,depth(320),[3,3],stride=2,padding="VALID",scope="a_conv2d_2")

                with tf.variable_scope('Branch_1'):
                    branch_1 = sipconv2d.ops(net,depth(192),[1,1],scope="b_conv2d_1")
                    branch_1 = sipconv2d.ops(branch_1, depth(192), [1, 7], scope='b_conv2d_2')
                    branch_1 = sipconv2d.ops(branch_1, depth(192), [7, 1], scope='b_conv2d_3')
                    branch_1 = sipconv2d.ops(branch_1, depth(192), [3, 3], stride=2,padding="VALID",scope='b_conv2d_4')

                with tf.variable_scope('Branch_2'):
                    branch_2 = layers_lib.max_pool2d(net,[3,3],stride=2,padding="VALID",scope="c_Max_Pool")
                net = tf.concat([branch_0, branch_1, branch_2], 3)
            # ----
            self.key_point[end_point] = net
            if end_point == self.final_endpoint:
                self.get_variables()
                return net
            # ---------------------------------------
            end_point = "Mixed_p10_b"
            # ---- ops
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = sipconv2d.ops(net, depth(320), [1, 1], scope="a_conv2d")

                with tf.variable_scope('Branch_1'):
                    branch_1 = sipconv2d.ops(net, depth(384), [1, 1], scope="b_conv2d_1")
                    branch_1 = tf.concat(
                        [
                            sipconv2d.ops(
                                branch_1, depth(384), [1, 3], scope='b_conv2d_2'),
                            sipconv2d.ops(
                                branch_1, depth(384), [3, 1], scope='b_conv2d_3')
                        ],
                        3)

                with tf.variable_scope('Branch_2'):
                    branch_2 = sipconv2d.ops(net, depth(448), [1, 1], scope="c_conv2d_1")
                    branch_2 = sipconv2d.ops(branch_2, depth(384), [3, 3], scope="c_conv2d_2")
                    branch_2 = tf.concat(
                        [
                            sipconv2d.ops(
                                branch_2, depth(384), [1, 3], scope='c_conv2d_3'),
                            sipconv2d.ops(
                                branch_2, depth(384), [3, 1], scope='c_conv2d_4')
                        ],
                        3)

                with tf.variable_scope('Branch_3'):
                    branch_3 = layers_lib.avg_pool2d(net, [3, 3], stride=1, padding="SAME",scope='d_AvgPool')
                    branch_3 = sipconv2d.ops(branch_3, depth(192), [1, 1], scope='d_conv2d')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            # ----
            self.key_point[end_point] = net
            if end_point == self.final_endpoint:
                self.get_variables()
                return net
            # ---------------------------------------
            end_point = "Mixed_p10_c"
            # ---- ops
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = sipconv2d.ops(net, depth(320), [1, 1], scope="a_conv2d")

                with tf.variable_scope('Branch_1'):
                    branch_1 = sipconv2d.ops(net, depth(384), [1, 1], scope="b_conv2d_1")
                    branch_1 = tf.concat(
                        [
                            sipconv2d.ops(
                                branch_1, depth(384), [1, 3], scope='b_conv2d_2'),
                            sipconv2d.ops(
                                branch_1, depth(384), [3, 1], scope='b_conv2d_3')
                        ],
                        3)

                with tf.variable_scope('Branch_2'):
                    branch_2 = sipconv2d.ops(net, depth(448), [1, 1], scope="c_conv2d_1")
                    branch_2 = sipconv2d.ops(branch_2, depth(384), [3, 3], scope="c_conv2d_2")
                    branch_2 = tf.concat(
                        [
                            sipconv2d.ops(
                                branch_2, depth(384), [1, 3], scope='c_conv2d_3'),
                            sipconv2d.ops(
                                branch_2, depth(384), [3, 1], scope='c_conv2d_4')
                        ],
                        3)

                with tf.variable_scope('Branch_3'):
                    branch_3 = layers_lib.avg_pool2d(net, [3, 3], stride=1, padding="SAME",scope='d_AvgPool')
                    branch_3 = sipconv2d.ops(branch_3, depth(192), [1, 1], scope='d_conv2d')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            # ----
            self.key_point[end_point] = net
            if end_point == self.final_endpoint:
                self.get_variables()
                return net
            # ---------------------------------------
