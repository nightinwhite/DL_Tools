import tensorflow as tf
from tensorflow.python.platform import flags
from tensorflow.contrib.layers.python.layers import regularizers
from DL_Tools.My_Net.Inception_v3 import Inception_v3
from DL_Tools.My_Net.utils import *
from DL_Tools.My_Net.Attention_Decoder import Attention_Decoder

class Attention_Model(object):
    def __init__(self,net_input,net_label = None):
        self.Flags = flags.FLAGS
        self.net_input = net_input
        self.batch_size = self.Flags.batch_size
        self.net_label = net_label
        self.is_training = self.Flags.is_training
        self.num_views = self.Flags.num_views
        # CNN_model -------------------------------
        self.final_point = self.Flags.final_point
        self.depth_multiplier = self.Flags.depth_multiplier
        self.min_depth = self.Flags.min_depth
        self.cnn_variable_list = None

        # LSTM_cell -------------------------------
        self.num_lstm_units = self.Flags.num_lstm_units
        self.cell_clip = self.Flags.lstm_state_clip_value

        # attention_decoder -----------------------
        self.num_char_classes = self.Flags.number_of_class
        self.zero_label = tf.zeros([self.batch_size, self.Flags.number_of_class])
        self.seq_length = self.Flags.label_length
        self.num_heads = self.Flags.num_heads
        self.attn_vec_length = None #!!!!!

        # loss
        self.label_smooth = self.Flags.label_smooth

        pass

    def get_variables(self):
        all_variables = tf.global_variables()
        self.train_variables = []
        self.train_variables_number = 0
        for v in all_variables:
            if (self.scope is not None and self.scope in v.name)\
                    or (self.scope is None and "Attention_Model" in v.name):
                self.train_variables.append(v)
                tmp_num = 1
                for n in v.get_shape():
                    tmp_num *= n.value
                self.train_variables_number += tmp_num
        return self.train_variables

    def pool_views_fn(self, nets):
        with tf.variable_scope('pool_views_fn'):
            net = tf.reduce_max(nets,0)
            batch_size = net.get_shape().dims[0].value
            height = net.get_shape().dims[1].value
            width = net.get_shape().dims[2].value
            feature_size = net.get_shape().dims[3].value
            return tf.reshape(net, [batch_size, height, width, feature_size])

    def build_model(self,scope = None,reuse = False):
        self.scope = scope
        with tf.variable_scope(self.scope, "Attention_Model",reuse=reuse):
            print "building Inception_v3..."
            inception = Inception_v3(final_endpoint=self.final_point,
                                     depth_multiplier=self.depth_multiplier,
                                     min_depth=self.min_depth,
                                     is_training=self.is_training)

            views = tf.split(
                value=self.net_input, num_or_size_splits=self.num_views, axis=2)
            nets = [
                inception.ops(v, scope="Inception_v3", reuse=(i != 0))
                for i, v in enumerate(views)
                ]
            net = self.pool_views_fn(nets)
            print net.get_shape()
            self.cnn_variable_list = inception.train_variables
            self.cnn_saver = tf.train.Saver(self.cnn_variable_list)
            print "Done."
            print "building Attention_Decoder..."
            with tf.variable_scope("LSTM_Cell"):
                lstm_cell = tf.contrib.rnn.LSTMCell(self.num_lstm_units,
                                                    use_peepholes=False,
                                                    cell_clip=self.cell_clip,
                                                    state_is_tuple=True,
                                                    initializer=orthogonal_initializer)
            # softmax varibales
            with tf.variable_scope("softmax_layer"):
                softmax_regularizer = regularizers.l2_regularizer(self.Flags.weight_decay)
                self.softmax_w = tf.get_variable("softmax_w",
                                                 [self.num_lstm_units, self.num_char_classes],
                                                 initializer=orthogonal_initializer,
                                                 regularizer=softmax_regularizer)
                self.softmax_b = tf.get_variable("softmax_b",
                                                 [self.num_char_classes],
                                                 initializer=tf.zeros_initializer(),
                                                 regularizer=softmax_regularizer)
                self.char_logits = {}

            attention_decoder = Attention_Decoder(seq_length=self.seq_length,
                                                  rnn_cell=lstm_cell,
                                                  weight_decay=self.Flags.weight_decay,
                                                  loop_function=self.get_input,
                                                  num_heads=1
                                                  )
            res_outputs,res_states = attention_decoder.ops(net,is_training=self.is_training)
            self.attns = attention_decoder.attns
            print "Done"
            self.saver = tf.train.Saver()
            self.net_res = res_outputs
            self.pred_res = [tf.expand_dims(self.get_char_logits(res_outputs[i],i),dim=1) for i in xrange(self.seq_length)]
            self.pred_res = tf.concat(self.pred_res,1)
            self.get_variables()

    def get_char_logits(self,input,index):
        with tf.variable_scope("get_char_logits"):
            if index not in self.char_logits:
                self.char_logits[index] = tf.nn.xw_plus_b(input, self.softmax_w, self.softmax_b)
            return self.char_logits[index]

    def char_one_hot(self, logit):
        with tf.variable_scope("char_one_hot"):
            prediction = tf.argmax(logit, axis=1)
            return tf.one_hot(prediction, self.num_char_classes)

    def label_smoothing_regularization(self,chars_labels, weight=0.1):
        with tf.variable_scope("label_smoothing_regularization"):
            # print type(FLAGS.num_char_classes)
            # print FLAGS.num_char_classes
            pos_weight = 1.0 - weight
            neg_weight = weight / self.num_char_classes
            return chars_labels * pos_weight + neg_weight

    def get_input(self,prev,i):
        with tf.variable_scope("get_input"):
            if i == 0:
                return self.zero_label
            else:
                if self.net_label is None:
                    this_label = self.char_one_hot(self.get_char_logits(prev, i - 1))
                else:
                    this_label = self.net_label[:, i - 1, :]
                return this_label

    def get_loss(self,net_label):
        with tf.variable_scope('sequence_loss'):
            if self.is_training:
                labels_list = tf.unstack(self.label_smoothing_regularization(net_label),axis=1)
            else:
                labels_list = tf.unstack(net_label, axis=1)

            batch_size, seq_length, _ = net_label.shape.as_list()
            weights = tf.ones((batch_size, seq_length), dtype=tf.float32)
                # # Suppose that reject character is the last in the charset.
                # reject_char = tf.constant(
                #     self._params.num_char_classes - 1,
                #     shape=(batch_size, seq_length),
                #     dtype=tf.int64)
                # known_char = tf.not_equal(chars_labels, reject_char)
                # weights = tf.to_float(known_char)
            logits_list = tf.unstack(self.pred_res, axis=1)
            weights_list = tf.unstack(weights, axis=1)
            def my_softmax_cross_entropy_with_logits(target, logit):
                return tf.nn.softmax_cross_entropy_with_logits(labels = target, logits = logit)
            loss = tf.contrib.legacy_seq2seq.sequence_loss(
                logits_list,
                labels_list,
                weights_list,
                softmax_loss_function=my_softmax_cross_entropy_with_logits,
                average_across_timesteps=False#!!!!!
            )
            regularizer_loss = tf.losses.get_regularization_losses()
            total_loss = tf.reduce_mean(loss + regularizer_loss)
            return total_loss


    def get_attns(self):
        return self.attns
    def eval(self):
        char_logist = self.pred_res
        char_pred = tf.argmax(char_logist,axis=-1)
        return char_logist, char_pred

    def save_model(self,sess, save_name):
        self.saver.save(sess, save_name)

    def save_cnn_model(self,sess, save_name):
        self.cnn_saver.save(sess,save_name)

    def restore_model(self,sess, restore_name):
        self.saver.restore(sess,restore_name)

    def restore_cnn_model(self,sess, restore_name):
        self.cnn_saver.restore(sess, restore_name)