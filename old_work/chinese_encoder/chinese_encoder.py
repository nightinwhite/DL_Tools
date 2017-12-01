import tensorflow as tf
from tensorflow.python.platform import flags as flags
import tensorflow.contrib.slim as slim
from My_NN.My_Net.Resnet_Encoder import ResNet_Encoder
from My_NN.My_Net.Resnet_Decoder import ResNet_Decoder

class Chinese_Map_NN(object):

    def __init__(self,net_input,is_training):
        self.FLAGS = flags.FLAGS
        self.is_training = is_training
        self.encoder_final_point = self.FLAGS.encoder_final_endpoint
        self.decoder_final_point = self.FLAGS.decoder_final_endpoint
        self.units_for_class = self.FLAGS.units_for_class
        self.units_for_style = self.FLAGS.units_for_style
        self.net_input = net_input
        self.resnet_encoder = ResNet_Encoder(self.encoder_final_point, self.is_training)
        self.resnet_decoder = ResNet_Decoder(self.decoder_final_point, self.is_training)

    def build_model(self,reuse = False):
        self.scope = "Chinese_Map_NN"
        with tf.variable_scope(self.scope,reuse=reuse):
            net = self.resnet_encoder.ops(self.net_input)
            _,net_h,net_w,net_c = list(net.get_shape())
            with tf.variable_scope("Map_Part"):
                with tf.variable_scope("Class_Part"):
                    class_flatten_net = slim.flatten(net)
                    self.class_vector = slim.fully_connected(class_flatten_net,self.units_for_class)
                with tf.variable_scope("Style_Part"):
                    style_matrix = tf.matmul(tf.transpose(net,[0,3,1,2]) , tf.transpose(net,[0,3,2,1]))
                    style_flatten_net = slim.flatten(style_matrix)
                    self.style_vector = slim.fully_connected(style_flatten_net,self.units_for_style)
                combine_net = tf.concat([self.class_vector,self.style_vector],axis=1)
                combine_net = slim.fully_connected(combine_net,int(net_h*net_w*net_c))
                combine_net = tf.reshape(combine_net,[-1,int(net_h),int(net_w),int(net_c)])
            net = self.resnet_decoder.ops(combine_net)
            self.map_img = net
            self.get_variables()

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




