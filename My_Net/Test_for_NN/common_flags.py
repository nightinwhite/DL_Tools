#coding:utf-8
from tensorflow.python.platform import flags
Flags = flags.FLAGS

def define():
    flags.DEFINE_integer("batch_size",100 ,"如名")
    flags.DEFINE_boolean("is_training",True,"训练模式")
    flags.DEFINE_integer("net_image_width", 48, "网络所需输入图像的宽度")
    flags.DEFINE_integer("net_image_height", 48, "网络所需输入图像的高度")
    flags.DEFINE_integer("net_image_channel", 3, "网络所需输入图像的通道数")
    flags.DEFINE_integer("label_length", 1, "label序列的长度")
    flags.DEFINE_integer("number_of_class", 10 + 1, "分类的数量（包含填充类）")

    flags.DEFINE_float("weight_decay", 0.00004, "规则化系数")
    flags.DEFINE_float("Conv_W_init_stddev", 0.01, "初始化系数")
    flags.DEFINE_float("depth_multiplier", 1.0, "CNN网络深度扩增系数")
    flags.DEFINE_integer("min_depth", 16, "CNN网络最小深度")
    flags.DEFINE_string("final_endpoint", "Mixed_p10_c", "Inception_v3的深度")
