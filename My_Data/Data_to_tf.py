import Queue
import threading
import numpy as np
import tensorflow as tf
from My_Log.Log_Manager import *


class Data_To_TF(object):
    def __init__(self,config,tf_path):
        self.config = config
        self.tf_path = tf_path
        self.min_of_queue = 10
        self.max_of_queue = 100


    def label_to_one_hot(self,label_index):
        num_of_class = len(self.config.char_to_label_dict)
        assert label_index < num_of_class
        res = np.full((num_of_class,0.))
        res[label_index] = 1.
        return res

    def parse_chars(self,chars):
        res = []
        for c in chars:
            res.append(self.config.char_to_label_dict[c])
        while len(res) < self.config.label_length:
            res.append(self.config.fill_label)
        return res

    def put_data(self):
        while self.read_index < len(self.data):
            # print self.data_queue.qsize()
            self.mutex.acquire()
            this_index = self.read_index
            self.read_index += 1
            self.mutex.release()
            tmp_data = self.data[this_index]
            real_data = None
            real_label = None
            if self.config.with_label:
                real_data = self.config.parse_data_fuc(tmp_data[0])
                real_label = self.parse_chars(tmp_data[1])
            else:
                real_data = self.config.parse_data_fuc(tmp_data)
            while self.data_queue.full() is True or this_index != self.put_index:
                pass
            self.mutex.acquire()
            if self.config.with_label:
                self.data_queue.put([real_data, real_label])
            else:
                self.data_queue.put(real_data)
            self.put_index += 1
            self.mutex.release()

    def generate(self):
        tf_writer = tf.python_io.TFRecordWriter(self.tf_path)
        self.data = self.config.read_data
        if self.config.is_shuffle:
            np.random.shuffle(self.data)
        data_len = len(self.data)
        print("data len:{0}".format(data_len))
        #
        self.read_index = 0
        self.put_index = 0
        self.mutex = threading.Lock()
        self.data_queue = Queue.Queue(self.max_of_queue)
        threads = []
        for t in xrange(self.config.thread_num):
            tmp_thread = threading.Thread(target=self.put_data)
            tmp_thread.setDaemon(True)
            tmp_thread.start()
            threads.append(tmp_thread)

        self.tmp_len = 0
        tmp_rate = 0
        while self.tmp_len < data_len:
            tmp_data = self.data_queue.get()
            if self.config.with_label:
                raw_image = tmp_data[0]
                tmp_lables = tmp_data[1]
            else:
                raw_image = tmp_data
            tmp_example = tf.train.Example(features=tf.train.Features(feature={
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw_image])),
                'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=tmp_lables)),
            }))
            tf_writer.write(tmp_example.SerializeToString())

            self.tmp_len += 1
            now_rate = int((self.tmp_len+0.)/data_len*100)
            if now_rate > tmp_rate:
                tmp_rate = now_rate
                print("generating : {0}%".format(tmp_rate))
        for t in threads:
            t.join()
        tf_writer.close()

