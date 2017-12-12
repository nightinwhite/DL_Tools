#coding:utf-8
import tensorflow as tf
import sys
import os
import time
import numpy as np
import Queue
import threading


class Info_Ops(object):
    def __init__(self,ops,ops_name,iter_num,visible = True,need_to_be_saved = False):
        self.ops = ops
        self.ops_name = ops_name
        assert iter_num > 0
        self.iter_num = iter_num
        self.visible = visible
        self.need_to_be_saved = need_to_be_saved

class Trainer(object):
    def __init__(self,sess,train_op,
                 epoch,train_iter_num,val_iter_num = 0
                 ):
        self.sess = sess
        self.train_op = train_op
        self.epoch = epoch
        self.train_iter_num = train_iter_num
        self.val_iter_num = val_iter_num
        self.train_info_ops_list = [] #训练时需要运行的节点
        self.val_info_ops_list = [] #验证时需要运行的节点
        # 模型存储
        self.save_path = None #模型保存的路径
        self.save_name = None #模型保存的名字
        # 最佳模型
        self.save_cmp_fuc = None #用于比较模型优劣的函数
        self.save_cmp_info = None #用于参与比较的信息
        self.save_cmp_now_value = None #用于参与比较的最新值
        # 观察随训练变化的节点
        self.visual_ops_list = [] #需要观察的节点
        self.visual_net_input_list = [] #需要填充数据的节点
        self.visual_real_input_list = [] #填充数据
        self.visual_save_fuc_list = [] #如何保存结果的函数
        # feed_dict
        self.get_train_feed_dict = lambda : None #用于返回训练时的feed_dict
        self.get_val_feed_dict = lambda : None #用于返回验证时的feed_dict
        # tf_event
        self.summary_writer = None #用于写tf_event
        # info_ops 保存用队列
        self.mutex = threading.Lock()
        self.ops_queue = Queue.Queue(self.train_iter_num + self.val_iter_num)
        self.is_training_finished = False
        self.tread_num = 4



    def add_train_info_ops(self,ops):
        self.train_info_ops_list.append(ops)

    def add_val_info_ops(self,ops):
        self.val_info_ops_list.append(ops)

    def add_visual_info_ops(self,ops,net_input,real_input,save_fuc):
        self.visual_ops_list.append(ops)
        self.visual_net_input_list.append(net_input)
        self.visual_real_input_list.append(real_input)
        self.visual_save_fuc_list.append(save_fuc)

    def set_model_saving_info(self,save_path,save_name):
        self.save_path = save_path
        self.save_name = save_name

    def set_best_saving_info(self,cmp_info_name,cmp_fuc):
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
    def saving_thread(self):
        while(self.is_training_finished == False or
                      self.ops_queue.empty() == False):
            self.mutex.acquire()
            if self.ops_queue.empty() == False:
                tmp_paras = self.ops_queue.get()
                self.mutex.release()
                tmp_flag = tmp_paras[0] # 标记批次
                tmp_epoch = tmp_flag.split("_")[0]
                tmp_itir = tmp_flag.split("_")[1]
                tmp_sess_all_res = tmp_paras[1]
                tmp_all_name = tmp_paras[2]
                tmp_all_need_save = tmp_paras[3]
                parents_path = self.info_ops_save_path
                tmp_save_path = os.path.join(parents_path,"{0}/".format(tmp_epoch))
                if os.path.exists(tmp_save_path) is False:
                    os.mkdir(tmp_save_path)
                for i in range(len(tmp_sess_all_res)):
                    tmp_value = tmp_sess_all_res[i]
                    tmp_name = tmp_all_name[i]
                    tmp_name.replace("/","%")
                    tmp_need_save = tmp_all_need_save[i]
                    if tmp_need_save == True:
                        tmp_save_name = os.path.join(tmp_save_path,"{0}_{1}.npy".format(tmp_itir,tmp_name))
                        np.save(tmp_save_name,tmp_value)
            else:
                self.mutex.release()

    def run(self):
        self.info_ops_save_path = self.save_name + "_save_info"
        if os.path.exists(self.info_ops_save_path) == False:
            os.mkdir(self.info_ops_save_path)
        saving_threads = []

        for i in range(self.tread_num):
            tmp_thread = threading.Thread(target=self.saving_thread)
            tmp_thread.setDaemon(True)
            tmp_thread.start()
            saving_threads.append(tmp_thread)

        for e in range(self.epoch):
            print "epoch:{0}/{1}".format(e+1,self.epoch)
            train_info_ops_now_value_dict = {}#""
            train_info_ops_now_num_dict = {}
            for i in range(self.train_iter_num):
                sess_run_list = [self.train_op]
                run_info_ops_name = []
                run_info_ops_visible_list = []
                run_info_ops_need_save_list = []
                for ops in self.train_info_ops_list:
                    if (e*self.train_iter_num + i)%ops.iter_num == 0:
                        sess_run_list.append(ops.ops)
                        run_info_ops_name.append(ops.ops_name)
                        run_info_ops_visible_list.append(ops.visible)
                        run_info_ops_need_save_list.append(ops.need_to_be_saved)
                sess_res = self.sess.run(sess_run_list,feed_dict = self.get_train_feed_dict())
                sess_res = sess_res[1:]
                train_info_print_line = "{0}/{1}: ".format(i+1, self.train_iter_num)
                self.mutex.acquire()
                self.ops_queue.put(["E{0}_I{1}_train".format(e,i),sess_res,run_info_ops_name,run_info_ops_need_save_list])
                self.mutex.release()
                for j in range(len(sess_res)):
                    tmp_value = sess_res[j]
                    tmp_name = run_info_ops_name[j]
                    tmp_visible = run_info_ops_visible_list[j]
                    # print tmp_name,tmp_visible
                    if tmp_visible:
                        if train_info_ops_now_value_dict.has_key(tmp_name):
                            bef_value = train_info_ops_now_value_dict[tmp_name]
                            bef_num = train_info_ops_now_value_dict[tmp_name]
                            tmp_value = (bef_value * bef_num + tmp_value) / (bef_num + 1)
                            train_info_ops_now_num_dict[tmp_name] = bef_num + 1
                            train_info_ops_now_value_dict[tmp_name] = tmp_value
                        else:
                            train_info_ops_now_num_dict[tmp_name] = 1
                            train_info_ops_now_value_dict[tmp_name] = tmp_value
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
                self.info_value_dict = train_info_ops_now_value_dict
            val_info_print_line = ""
            if self.val_iter_num != 0:
                val_info_ops_now_value_dict = {}  # ""
                val_info_ops_now_num_dict = {}
                for i in range(self.val_iter_num):
                    if len(self.val_info_ops_list) == 0:
                        break
                    sess_run_list = []
                    run_info_ops_name = []
                    run_info_ops_need_save_list = []
                    for ops in self.val_info_ops_list:
                        if self.val_iter_num % ops.iter_num == 0:
                            sess_run_list.append(ops.ops)
                            run_info_ops_name.append(ops.ops_name)
                            run_info_ops_need_save_list.append(ops.need_to_be_saved)
                    sess_res = self.sess.run(sess_run_list,feed_dict = self.get_val_feed_dict())
                    # print run_info_ops_name, sess_res
                    self.mutex.acquire()
                    self.ops_queue.put(
                        ["E{0}_I{1}_val".format(e, i), sess_res, run_info_ops_name, run_info_ops_need_save_list])
                    self.mutex.release()
                    for j in range(len(sess_res)):
                        tmp_value = sess_res[j]
                        tmp_name = run_info_ops_name[j]
                        if val_info_ops_now_value_dict.has_key(tmp_name):
                            bef_value = val_info_ops_now_value_dict[tmp_name]
                            bef_num = val_info_ops_now_num_dict[tmp_name]
                            tmp_value = (bef_value * bef_num + tmp_value) / (bef_num + 1)
                            val_info_ops_now_num_dict[tmp_name] = bef_num + 1
                            val_info_ops_now_value_dict[tmp_name] = tmp_value
                        else:
                            val_info_ops_now_num_dict[tmp_name] = 1
                            val_info_ops_now_value_dict[tmp_name] = tmp_value
                    sys.stdout.write("                                  \r")
                    sys.stdout.flush()
                    sys.stdout.write("validating:{0}/{1}".format(i+1,self.val_iter_num))

                print ""
                for ops in self.val_info_ops_list:
                    val_info_print_line += "{0}:{1} ".format(ops.ops_name,val_info_ops_now_value_dict[ops.ops_name])
            print val_info_print_line
            self.saver = tf.train.Saver(max_to_keep=3)
            if self.save_path is None:
                if os.path.exists("models") == False:
                    os.mkdir("models")
                self.save_name = "model"
                self.save_path = "models"
            self.saver.save(self.sess,os.path.join(self.save_path,"{0}_{1}".format(self.save_name,"new")))
            if self.save_cmp_fuc is not None:
                for k in val_info_ops_now_value_dict.keys():
                    self.info_value_dict[k] = val_info_ops_now_value_dict[k]
            if self.save_cmp_fuc is not None:
                self.save_best()

            while self.ops_queue.empty() == False:
                sys.stdout.write("                                  \r")
                sys.stdout.flush()
                sys.stdout.write("busy saving...leave {0} to save".format(self.ops_queue.qsize()))
            print ""
            for index in range(len(self.visual_ops_list)):
                tmp_ops = self.visual_ops_list[index]
                tmp_net_input = self.visual_net_input_list[index]
                tmp_real_input = self.visual_real_input_list[index]
                tmp_save_fuc = self.visual_save_fuc_list[index]
                tmp_res = self.sess.run(tmp_ops, feed_dict = {tmp_net_input : tmp_real_input})
                tmp_save_fuc(tmp_res,e)
        self.is_training_finished = True
        for t in saving_threads:
            t.join()





