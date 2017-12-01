#coding:utf-8

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
def define():
    flags.DEFINE_integer("batch_size", 16, "")
    flags.DEFINE_integer("net_img_width", 64, "")
    flags.DEFINE_integer("net_img_height", 64, "")
    flags.DEFINE_integer("net_img_channel", 3, "")

    flags.DEFINE_integer("epoch", 5000, "批次数量")
    flags.DEFINE_integer("train_iter_num", 400, "每批次迭代数量")
    flags.DEFINE_integer("val_iter_num", 10, "每批次验证迭代数量")

    flags.DEFINE_integer("units_for_class", 2048, "类别vector")
    flags.DEFINE_integer("units_for_style", 2048, "风格vector")
    flags.DEFINE_float("weight_decay", 0.00004, "规则化系数")
    flags.DEFINE_float("Conv_W_init_stddev", 0.1, "初始化系数")
    flags.DEFINE_float("map_factor",   0.8, "map loss影响系数")
    flags.DEFINE_float("class_factor", 0.1, "class loss影响系数")
    flags.DEFINE_float("style_factor", 0.1, "style loss影响系数")

    flags.DEFINE_float("learning_rate", 0.001, "learning_rate")
    flags.DEFINE_float("momentum", 0.9, "momentum")
    flags.DEFINE_float("clip_gradient_norm", 2.0, "clip_gradient_norm")
    flags.DEFINE_float('label_smoothing', 0.1,
                       'weight for label smoothing')
    flags.DEFINE_string("encoder_final_endpoint","Block{0}_O_{1}_C_{2}_S_{3}",'Endpoint to Encoder')
    flags.DEFINE_string("decoder_final_endpoint", "D_Block_{0}_O_{1}_C_{2}_S_{3}", 'Endpoint to Dncoder')
