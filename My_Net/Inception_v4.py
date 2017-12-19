#coding:utf-8
import tensorflow as tf
import tensorflow.contrib.slim as slim

class Inception_v4(object):
    def __init__(self,final_endpoint,Conv_W_init_stddev,weight_decay,is_training):
        self.final_endpoint = final_endpoint
        self.Conv_W_init_stddev = Conv_W_init_stddev
        self.weight_decay = weight_decay
        self.is_training = is_training
        self.key_point = {}

    def conv2d_bn(self,x, nb_filter, num_row, num_col,scope,
                  padding='same', strides=(1, 1), use_bias=False):
        with tf.variable_scope(scope):
            x = tf.layers.conv2d(x, nb_filter, (num_row, num_col), strides=strides, padding=padding,
                                 use_bias=use_bias,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=self.Conv_W_init_stddev),
                                 kernel_regularizer=slim.l2_regularizer(self.weight_decay))

            x = slim.batch_norm(x, decay=0.9997, scale=False)
            x = tf.nn.relu(x)
            return x

    def block_inception_a(self, input, scope):
        with tf.variable_scope(scope):
            branch_0 = self.conv2d_bn(input, 96, 1, 1,"branch_0_conv2d")

            branch_1 = self.conv2d_bn(input, 64, 1, 1,"branch_1_conv2d_0")
            branch_1 = self.conv2d_bn(branch_1, 96, 3, 3,"branch_1_conv2d_1")

            branch_2 = self.conv2d_bn(input, 64, 1, 1,"branch_2_conv2d_0")
            branch_2 = self.conv2d_bn(branch_2, 96, 3, 3,"branch_2_conv2d_1")
            branch_2 = self.conv2d_bn(branch_2, 96, 3, 3,"branch_2_conv2d_2")

            branch_3 = tf.layers.average_pooling2d(input,(3, 3), strides=(1, 1), padding='same')
            branch_3 = self.conv2d_bn(branch_3, 96, 1, 1,"branch_3_conv2d")

            x = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=4)
            return x

    def block_reduction_a(self, input, scope):
        with tf.variable_scope(scope):
            branch_0 = self.conv2d_bn(input, 384, 3, 3, strides=(2, 2), padding='valid',scope="branch_0_conv2d")

            branch_1 = self.conv2d_bn(input, 192, 1, 1, "branch_1_conv2d_0")
            branch_1 = self.conv2d_bn(branch_1, 224, 3, 3, "branch_1_conv2d_1")
            branch_1 = self.conv2d_bn(branch_1, 256, 3, 3, strides=(2, 2), padding='valid',scope="branch_1_conv2d_2")

            branch_2 = tf.layers.max_pooling2d(input,(3, 3), strides=(2, 2), padding='valid')

            x = tf.concat([branch_0, branch_1, branch_2], axis=4)
            return x

    def block_inception_b(self, input, scope):
        with tf.variable_scope(scope):
            branch_0 = self.conv2d_bn(input, 384, 1, 1,"branch_0_conv2d")

            branch_1 = self.conv2d_bn(input, 192, 1, 1,"branch_1_conv2d_0")
            branch_1 = self.conv2d_bn(branch_1, 224, 1, 7,"branch_1_conv2d_1")
            branch_1 = self.conv2d_bn(branch_1, 256, 7, 1,"branch_1_conv2d_2")

            branch_2 = self.conv2d_bn(input, 192, 1, 1,"branch_2_conv2d_0")
            branch_2 = self.conv2d_bn(branch_2, 192, 7, 1,"branch_2_conv2d_1")
            branch_2 = self.conv2d_bn(branch_2, 224, 1, 7,"branch_2_conv2d_2")
            branch_2 = self.conv2d_bn(branch_2, 224, 7, 1,"branch_2_conv2d_3")
            branch_2 = self.conv2d_bn(branch_2, 256, 1, 7,"branch_2_conv2d_4")

            branch_3 = tf.layers.average_pooling2d(input,(3, 3), strides=(1, 1), padding='same')
            branch_3 = self.conv2d_bn(branch_3, 128, 1, 1,"branch_3_conv2d")

            x = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=-1)
            return x

    def block_reduction_b(self, input, scope):
        with tf.variable_scope(scope):
            branch_0 = self.conv2d_bn(input, 192, 1, 1,"branch_0_conv2d_0")
            branch_0 = self.conv2d_bn(branch_0, 192, 3, 3,"branch_0_conv2d_1", strides=(2, 2), padding='valid')

            branch_1 = self.conv2d_bn(input, 256, 1, 1,"branch_1_conv2d_0")
            branch_1 = self.conv2d_bn(branch_1, 256, 1, 7,"branch_1_conv2d_1")
            branch_1 = self.conv2d_bn(branch_1, 320, 7, 1,"branch_1_conv2d_2")
            branch_1 = self.conv2d_bn(branch_1, 320, 3, 3,"branch_1_conv2d_3", strides=(2, 2), padding='valid')

            branch_2 = tf.layers.max_pooling2d(input,(3, 3), strides=(2, 2), padding='valid')

            x = tf.concat([branch_0, branch_1, branch_2], axis=-1)
            return x

    def block_inception_c(self, input, scope):

        branch_0 = self.conv2d_bn(input, 256, 1, 1,"branch_0_conv2d")

        branch_1 = self.conv2d_bn(input, 384, 1, 1,"branch_1_conv2d")
        branch_10 = self.conv2d_bn(branch_1, 256, 1, 3,"branch_10_conv2d")
        branch_11 = self.conv2d_bn(branch_1, 256, 3, 1,"branch_11_conv2d")
        branch_1 = tf.concat([branch_10, branch_11], axis=-1)

        branch_2 = self.conv2d_bn(input, 384, 1, 1,"branch_2_conv2d_0")
        branch_2 = self.conv2d_bn(branch_2, 448, 3, 1,"branch_2_conv2d_1")
        branch_2 = self.conv2d_bn(branch_2, 512, 1, 3,"branch_2_conv2d_2")
        branch_20 = self.conv2d_bn(branch_2, 256, 1, 3,"branch_20_conv2d")
        branch_21 = self.conv2d_bn(branch_2, 256, 3, 1,"branch_21_conv2d")
        branch_2 = tf.concat([branch_20, branch_21], axis=-1)

        branch_3 = tf.layers.average_pooling2d((3, 3), strides=(1, 1), padding='same')
        branch_3 = self.conv2d_bn(branch_3, 256, 1, 1,"branch_3_conv2d")
        x = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=-1)
        return x

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


    def __call__(self,input,scope=None,reuse = False):
        tmp_key_point_name = "p1_conv2d"

        net = self.conv2d_bn(input, 32, 3, 3,"p1_conv2d", strides=(2, 2), padding='valid')
        # --------------------------------------------
        self.key_point[tmp_key_point_name] = net
        if tmp_key_point_name == self.final_endpoint:
            return net
        tmp_key_point_name = "p2_conv2d"
        #---------------------------------------------
        net = self.conv2d_bn(net, 32, 3, 3,"p2_conv2d", padding='valid')
        # --------------------------------------------
        self.key_point[tmp_key_point_name] = net
        if tmp_key_point_name == self.final_endpoint:
            return net
        tmp_key_point_name = "p3_conv2d"
        # ---------------------------------------------
        net = self.conv2d_bn(net, 64, 3, 3,"p3_conv2d",)
        # --------------------------------------------
        self.key_point[tmp_key_point_name] = net
        if tmp_key_point_name == self.final_endpoint:
            return net
        tmp_key_point_name = "p4_mixed"
        # ---------------------------------------------
        branch_0 = tf.layers.max_pooling2d(net,(3, 3), strides=(2, 2), padding='valid')

        branch_1 = self.conv2d_bn(net, 96, 3, 3, "p4_conv2d",strides=(2, 2), padding='valid')

        net = tf.concat([branch_0, branch_1], axis=-1)
        # --------------------------------------------
        self.key_point[tmp_key_point_name] = net
        if tmp_key_point_name == self.final_endpoint:
            return net
        tmp_key_point_name = "p5_mixed"
        # ---------------------------------------------
        branch_0 = self.conv2d_bn(net, 64, 1, 1, "p5_conv2d_branch0_0")
        branch_0 = self.conv2d_bn(branch_0, 96, 3, 3, "p5_conv2d_branch0_1", padding='valid')

        branch_1 = self.conv2d_bn(net, 64, 1, 1, "p5_conv2d_branch1_0")
        branch_1 = self.conv2d_bn(branch_1, 64, 1, 7, "p5_conv2d_branch1_1")
        branch_1 = self.conv2d_bn(branch_1, 64, 7, 1, "p5_conv2d_branch1_2")
        branch_1 = self.conv2d_bn(branch_1, 96, 3, 3, "p5_conv2d_branch1_3", padding='valid')

        net = tf.concat([branch_0, branch_1], axis=-1)
        # --------------------------------------------
        self.key_point[tmp_key_point_name] = net
        if tmp_key_point_name == self.final_endpoint:
            return net
        tmp_key_point_name = "p6_mixed"
        # ---------------------------------------------
        branch_0 = self.conv2d_bn(net, 192, 3, 3, "p6_conv2d_branch0", strides=(2, 2), padding='valid')
        branch_1 = tf.layers.max_pooling2d(net,(3, 3), strides=(2, 2), padding='valid')

        net = tf.concat([branch_0, branch_1], axis=-1)
        # --------------------------------------------
        self.key_point[tmp_key_point_name] = net
        if tmp_key_point_name == self.final_endpoint:
            return net
        tmp_key_point_name = "p7_mixed_0"
        # ---------------------------------------------
        # 35 x 35 x 384
        # 4 x Inception-A blocks
        for idx in range(4):
            net = self.block_inception_a(net,"p7_mixed_{0}".format(idx))
            # --------------------------------------------
            self.key_point[tmp_key_point_name] = net
            if tmp_key_point_name == self.final_endpoint:
                return net
            tmp_key_point_name = "p7_mixed_{0}".format(idx+1)
            # ---------------------------------------------
        # 35 x 35 x 384
        # Reduction-A block
        tmp_key_point_name = "p7_mixed_reduction"
        net = self.block_reduction_a(net,"p7_mixed_reduction")
        # --------------------------------------------
        self.key_point[tmp_key_point_name] = net
        if tmp_key_point_name == self.final_endpoint:
            return net
        tmp_key_point_name = "p8_mixed_0"
        # ---------------------------------------------
        # 17 x 17 x 1024
        # 7 x Inception-B blocks
        for idx in range(7):
            net = self.block_inception_b(net,"p8_mixed_{0}".format(idx))
            # --------------------------------------------
            self.key_point[tmp_key_point_name] = net
            if tmp_key_point_name == self.final_endpoint:
                return net
            tmp_key_point_name = "p8_mixed_{0}".format(idx+1)
            # ---------------------------------------------
        # 17 x 17 x 1024
        # Reduction-B block
        tmp_key_point_name = "p8_mixed_reduction"
        net = self.block_reduction_b(net,"p8_mixed_reduction")
        # --------------------------------------------
        self.key_point[tmp_key_point_name] = net
        if tmp_key_point_name == self.final_endpoint:
            return net
        tmp_key_point_name = "p9_mixed_0"
        # ---------------------------------------------
        # 8 x 8 x 1536
        # 3 x Inception-C blocks
        for idx in range(3):
            net = self.block_inception_c(net,"p9_mixed_{0}".format(idx))
            # --------------------------------------------
            self.key_point[tmp_key_point_name] = net
            if tmp_key_point_name == self.final_endpoint:
                return net
            tmp_key_point_name = "p9_mixed_{0}".format(idx+1)
            # ---------------------------------------------


