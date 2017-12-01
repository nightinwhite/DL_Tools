import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys
import os
import time
import numpy as np
class Info_Ops(object):
    def __init__(self,ops,ops_name,iter_num,visible = True):
        self.ops = ops
        self.ops_name = ops_name
        assert iter_num > 0
        self.iter_num = iter_num
        self.visible = visible

class Trainer(object):
    def __init__(self,sess,train_loss,optimizer_name,learning_rate,
                 epoch,train_iter_num,val_iter_num = 0
                 ):
        self.sess = sess
        self.train_loss = train_loss
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self._get_train_op()
        self.epoch = epoch
        self.train_iter_num = train_iter_num
        self.val_iter_num = val_iter_num
        self.train_info_ops_list = []
        self.val_info_ops_list = []
        self.save_info_dict = {}

        self.save_path = None
        self.save_name = None

        self.save_cmp_fuc = None
        self.save_cmp_info = None
        self.save_cmp_now_value = None

        self.visual_ops_list = []
        self.visual_net_input_list = []
        self.visual_real_input_list = []
        self.visual_save_fuc_list = []

        self.get_train_feed_dict = lambda : None
        self.get_val_feed_dict = lambda : None

        self.summary_writer = None


    def _get_train_op(self,):
        if self.optimizer_name == "MomentumOptimizer":
            self.optimizer = tf.train.MomentumOptimizer(self.learning_rate,momentum=0.9)
        if self.optimizer_name == "AdamOptimizer":
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)


    def add_train_info_ops(self,ops):
        self.train_info_ops_list.append(ops)

    def add_val_info_ops(self,ops):
        self.val_info_ops_list.append(ops)

    def add_visual_info(self,info_name):
        self.save_info_dict[info_name] = []

    def set_clip_value(self,clip_value):
        with tf.variable_scope("clip_gradient_norm"):
            self.train_op = slim.learning.create_train_op(self.train_loss, self.optimizer,
                                                  clip_gradient_norm=clip_value)
    def add_visual_info_ops(self,ops,net_input,real_input,save_fuc):
        self.visual_ops_list.append(ops)
        self.visual_net_input_list.append(net_input)
        self.visual_real_input_list.append(real_input)
        self.visual_save_fuc_list.append(save_fuc)

    def set_save_tf_info(self,save_path,save_name):
        self.save_path = save_path
        self.save_name = save_name

    def set_best_save_info(self,cmp_info_name,cmp_fuc):
        self.save_cmp_fuc = cmp_fuc
        self.save_cmp_info_name = cmp_info_name

    def set_get_train_feed_dict(self, get_feed_dic_fuc):
        self.get_train_feed_dict = get_feed_dic_fuc

    def set_get_val_feed_dict(self, get_feed_dic_fuc):
        self.get_val_feed_dict = get_feed_dic_fuc

    def set_summary_writer(self,merge_summary,summary_writer,iter_num):
        self.merge_summary = merge_summary
        self.summary_writer = summary_writer
        self.summary_iter_num = iter_num

    def save_best(self):
        if self.save_cmp_now_value is None:
            self.save_cmp_now_value = self.info_value_dict[self.save_cmp_info_name]
            self.saver.save(self.sess,os.path.join(self.save_path,
                                                   "{0}_{1}_{2}".format(self.save_name,self.save_cmp_info_name,
                                                                        self.save_cmp_now_value)))
        else:
            if self.save_cmp_fuc(self.save_cmp_now_value,self.info_value_dict[self.save_cmp_info_name]):
                self.save_cmp_now_value = self.info_value_dict[self.save_cmp_info_name]
                self.saver.save(self.sess, os.path.join(self.save_path,
                                                        "{0}_{1}_{2}".format(self.save_name, self.save_cmp_info_name,
                                                                             self.save_cmp_now_value)))
    def init_model(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        tf.train.start_queue_runners(self.sess)

    def run(self):
        if len(self.save_info_dict.keys())!=0:
            self.save_info_save_path = self.save_name + "_save_info"
            if os.path.exists(self.save_info_save_path) == False:
                os.mkdir(self.save_info_save_path)
        for e in range(self.epoch):
            print "epoch:{0}/{1}".format(e+1,self.epoch)
            train_info_now_value_dict = {}#""
            train_info_now_num_dict = {}
            for i in range(self.train_iter_num):
                sess_run_list = [self.train_op,self.train_loss]
                run_info_ops_name = ["train_loss"]
                run_info_visible_list = [True]
                for ops in self.train_info_ops_list:
                    if (e*self.train_iter_num + self.train_iter_num)%ops.iter_num == 0:
                        sess_run_list.append(ops.ops)
                        run_info_ops_name.append(ops.ops_name)
                        run_info_visible_list.append(ops.visible)
                sess_res = self.sess.run(sess_run_list,feed_dict = self.get_train_feed_dict())
                sess_res = sess_res[1:]
                train_info_print_line = "{0}/{1}: ".format(i+1, self.train_iter_num)
                for j in range(len(sess_res)):
                    tmp_value = sess_res[j]
                    tmp_name = run_info_ops_name[j]
                    if tmp_name in self.save_info_dict.keys():
                        bef_save_info = self.save_info_dict[tmp_name]
                        bef_save_info.append(tmp_value)
                        now_save_info = bef_save_info
                        self.save_info_dict[tmp_name] = now_save_info

                    if train_info_now_value_dict.has_key(tmp_name):
                        bef_value = train_info_now_value_dict[tmp_name]
                        bef_num = train_info_now_num_dict[tmp_name]
                        tmp_value = (bef_value*bef_num + tmp_value)/(bef_num+1)
                        train_info_now_num_dict[tmp_name] = bef_num + 1
                        train_info_now_value_dict[tmp_name] = tmp_value
                    else:
                        train_info_now_num_dict[tmp_name] = 1
                        train_info_now_value_dict[tmp_name] = tmp_value
                    if run_info_visible_list[j] == True:
                        train_info_print_line += "{0}:{1} ".format(tmp_name,tmp_value)
                sys.stdout.write(" "*len(train_info_print_line)*2 + "\r")
                sys.stdout.flush()
                sys.stdout.write(train_info_print_line)

                if self.summary_writer is not None:
                    if i % self.summary_iter_num == 0:
                        summary_data = self.sess.run(self.merge_summary,feed_dict = self.get_train_feed_dict())
                        self.summary_writer.add_summary(summary_data,e*self.epoch+i)
            print ""
            self.info_value_dict = {}
            if self.save_cmp_fuc is not None:
                self.info_value_dict = train_info_now_value_dict

            if self.val_iter_num != 0:
                val_info_now_value_dict = {}  # ""
                val_info_now_num_dict = {}
                for i in range(self.val_iter_num):
                    if len(self.val_info_ops_list) == 0:
                        break
                    sess_run_list = []
                    run_info_ops_name = []
                    for ops in self.val_info_ops_list:
                        if self.val_iter_num % ops.iter_num == 0:
                            sess_run_list.append(ops.ops)
                            run_info_ops_name.append(ops.ops_name)

                    sess_res = self.sess.run(sess_run_list,feed_dict = self.get_val_feed_dict())
                    # print run_info_ops_name, sess_res
                    for j in range(len(sess_res)):
                        tmp_value = sess_res[j]
                        tmp_name = run_info_ops_name[j]
                        if tmp_name in self.save_info_dict.keys():
                            bef_save_info = self.save_info_dict[tmp_name]
                            bef_save_info.append(tmp_value)
                            now_save_info = bef_save_info
                            self.save_info_dict[tmp_name] = now_save_info

                        if val_info_now_value_dict.has_key(tmp_name):
                            bef_value = val_info_now_value_dict[tmp_name]
                            bef_num = val_info_now_num_dict[tmp_name]
                            tmp_value = (bef_value * bef_num + tmp_value) / (bef_num + 1)
                            val_info_now_num_dict[tmp_name] = bef_num + 1
                            val_info_now_value_dict[tmp_name] = tmp_value
                        else:
                            val_info_now_num_dict[tmp_name] = 1
                            val_info_now_value_dict[tmp_name] = tmp_value
                    sys.stdout.write("                                  \r")
                    sys.stdout.flush()
                    sys.stdout.write("validating:{0}/{1}".format(i+1,self.val_iter_num))
                val_info_print_line = ""
                print ""
                for ops in self.val_info_ops_list:
                    val_info_print_line += "{0}:{1} ".format(ops.ops_name,val_info_now_value_dict[ops.ops_name])
            print val_info_print_line
            self.saver = tf.train.Saver(max_to_keep=3)
            if self.save_path is None:
                if os.path.exists("models") == False:
                    os.mkdir("models")
                self.save_name = "model"
                self.save_path = "models"
            self.saver.save(self.sess,os.path.join(self.save_path,"{0}_{1}".format(self.save_name,"new")))
            if self.save_cmp_fuc is not None:
                for k in val_info_now_value_dict.keys():
                    self.info_value_dict[k] = val_info_now_value_dict[k]
            if self.save_cmp_fuc is not None:
                self.save_best()

            for k in self.save_info_dict.keys():
                tmp_list = self.save_info_dict[k]
                if len(tmp_list) != 0:
                    tmp_path = os.path.join(self.save_info_save_path,k)
                    if os.path.exists(tmp_path) == False:
                        os.mkdir(tmp_path)
                    tmp_name = "e_{0}_t_{1}_.npy".format(e,time.time())
                    tmp_save_path = os.path.join(tmp_path,tmp_name)
                    tmp_list = np.asarray(tmp_list)
                    np.save(tmp_save_path,tmp_list)
                    self.save_info_dict[k] = []

            for index in range(len(self.visual_ops_list)):
                tmp_ops = self.visual_ops_list[index]
                tmp_net_input = self.visual_net_input_list[index]
                tmp_real_input = self.visual_real_input_list[index]
                tmp_save_fuc = self.visual_save_fuc_list[index]
                tmp_res = self.sess.run(tmp_ops, feed_dict = {tmp_net_input : tmp_real_input})
                tmp_save_fuc(tmp_res,e)





