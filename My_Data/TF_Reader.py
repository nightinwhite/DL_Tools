import tensorflow as tf
from tensorflow.python.platform import flags

class TF_Reader(object):
    def __init__(self,file_paths,with_label):

        self.with_label = with_label
        self.batch_num_thread = 4
        Flags = flags.FLAGS
        self.batch_size = Flags.batch_size
        self.batch_capacity = 4 * self.batch_size
        self.batch_min_num = 2 * self.batch_size
        self.label_length = Flags.label_length
        self.num_of_class = Flags.number_of_class
        self.H_W_C = [Flags.net_image_height,Flags.net_image_width,Flags.net_image_channel]

        if with_label:
            assert self.label_length != 0
            assert self.num_of_class != 0
        filename_queue = tf.train.string_input_producer(file_paths)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        if with_label:
            feartures = tf.parse_single_example(serialized_example,
                                                features={
                                                    'img_raw': tf.FixedLenFeature([], tf.string),
                                                    'labels': tf.FixedLenFeature([self.label_length], tf.int64),

                                                }
                                                )
        else:
            feartures = tf.parse_single_example(serialized_example,
                                                features={
                                                    'img_raw': tf.FixedLenFeature([], tf.string),
                                                }
                                                )
        self.img = feartures["img_raw"]
        self.img = tf.decode_raw(self.img, tf.uint8)
        self.img = tf.cast(self.img, tf.float32)
        self.img = tf.reshape(self.img, self.H_W_C)
        if with_label:
            self.label = feartures["labels"]
            self.label_one_hot = tf.one_hot(self.label,self.num_of_class)

    def data_argument(self,arg_ops):
        for op in arg_ops:
            self.img = op(self.img)

    def shuffle_batch(self):
        if self.with_label:
            return tf.train.shuffle_batch([self.img,self.label_one_hot],self.batch_size,
                                          capacity=self.batch_capacity,
                                          min_after_dequeue=self.batch_min_num,
                                          num_threads=self.batch_num_thread)
        else:
            tmp_res = tf.train.shuffle_batch([self.img], self.batch_size,
                                          capacity=self.batch_capacity,
                                          min_after_dequeue=self.batch_min_num,
                                          num_threads=self.batch_num_thread)
            return tmp_res[0]

    def batch(self):
        if self.with_label:
            return tf.train.batch([self.img,self.label_one_hot],self.batch_size,
                                  capacity=self.batch_capacity)

        else:
            tmp_res = tf.train.batch([self.img], self.batch_size,
                                          capacity=self.batch_capacity,)
            return tmp_res[0]


