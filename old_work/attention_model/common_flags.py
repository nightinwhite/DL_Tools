#coding:utf-8
from tensorflow.python.platform import flags
Flags = flags.FLAGS
# TF_reader
def define():
    flags.DEFINE_integer("batch_size",32 ,"如名")
    flags.DEFINE_boolean("is_training",True,"训练模式")
    flags.DEFINE_integer("net_image_width", 200, "网络所需输入图像的宽度")
    flags.DEFINE_integer("net_image_height", 70, "网络所需输入图像的高度")
    flags.DEFINE_integer("net_image_channel", 3, "网络所需输入图像的通道数")
    flags.DEFINE_integer("label_length", 8, "label序列的长度")
    flags.DEFINE_integer("number_of_class", 61 + 1, "分类的数量（包含填充类）")
    flags.DEFINE_integer("num_views", 1, "图片的角度数量")

    flags.DEFINE_float("weight_decay", 0.00004, "规则化系数")
    flags.DEFINE_float("Conv_W_init_stddev", 0.1, "初始化系数")
    flags.DEFINE_float("depth_multiplier", 1.0, "CNN网络深度扩增系数")
    flags.DEFINE_integer("min_depth", 16, "CNN网络最小深度")
    flags.DEFINE_string("final_point","Mixed_p8_c","Inception_v3的深度")

    flags.DEFINE_float("label_smooth", 0.1, "训练时label的处理")
    flags.DEFINE_integer("num_lstm_units", 256, "lstm单元数")
    flags.DEFINE_float('lstm_state_clip_value', 10.0,"防止梯度爆炸")

    flags.DEFINE_integer("num_heads", 1, "attention mask 数量")

    flags.DEFINE_float("learning_rate", 0.004, "learning_rate")
    flags.DEFINE_float("momentum", 0.9, "momentum")
    flags.DEFINE_float("clip_gradient_norm", 2.0, "防止梯度爆炸")

    flags.DEFINE_integer("epoch", 1000, "批次数量")
    flags.DEFINE_integer("train_iter_num", 1000, "每批次迭代数量")
    flags.DEFINE_integer("val_iter_num", 10, "每批次验证迭代数量")
