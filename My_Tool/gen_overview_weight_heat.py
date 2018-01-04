from My_Tool.Image_Drawer import *
import os
from PIL import Image
import numpy as np
class overview_layer(Image_Drawer):
    def __init__(self,layer_name,super_layer,layer_idx,layer_shape,sub_direction,sub_interval,sub_x_bias,sub_y_bias):
        Image_Drawer.__init__(self,layer_name,super_layer,layer_idx,layer_shape,sub_direction,sub_interval,sub_x_bias,sub_y_bias)
        self.parent_dir = "visualization_graphs/weights_heat_analyse/"
        # self.class_dir = ["I0_Attention_Model%Inception_v3%Conv2d_p1_3x3%Conv%weights:0",
        #         "I0_Attention_Model%Inception_v3%Conv2d_p2_3x3%Conv%weights:0",
        #         "I0_Attention_Model%Inception_v3%Conv2d_p3_3x3%Conv%weights:0",
        #         "I0_Attention_Model%Inception_v3%Conv2d_p5_1x1%Conv%weights:0",
        #         "I0_Attention_Model%Inception_v3%Conv2d_p6_3x3%Conv%weights:0",
        #         "I0_Attention_Model%Inception_v3%Mixed_p8_a%Branch_0%a_conv2d%Conv%weights:0",
        #         "I0_Attention_Model%Inception_v3%Mixed_p8_a%Branch_1%b_conv2d_1%Conv%weights:0",
        #         "I0_Attention_Model%Inception_v3%Mixed_p8_a%Branch_1%b_conv2d_2%Conv%weights:0",
        #         "I0_Attention_Model%Inception_v3%Mixed_p8_a%Branch_2%c_conv2d_1%Conv%weights:0",
        #         "I0_Attention_Model%Inception_v3%Mixed_p8_a%Branch_2%c_conv2d_2%Conv%weights:0",
        #         "I0_Attention_Model%Inception_v3%Mixed_p8_a%Branch_2%c_conv2d_3%Conv%weights:0",
        #         "I0_Attention_Model%Inception_v3%Mixed_p8_a%Branch_2%c_conv2d_3%Conv%weights:0",
        #         "I0_Attention_Model%Inception_v3%Mixed_p8_a%Branch_3%d_conv2d%Conv%weights:0",
        #         "I0_Attention_Model%Inception_v3%Mixed_p8_b%Branch_0%a_conv2d%Conv%weights:0",
        #         "I0_Attention_Model%Inception_v3%Mixed_p8_b%Branch_1%b_conv2d_1%Conv%weights:0",
        #         "I0_Attention_Model%Inception_v3%Mixed_p8_b%Branch_1%b_conv2d_2%Conv%weights:0",
        #         "I0_Attention_Model%Inception_v3%Mixed_p8_b%Branch_2%c_conv2d_1%Conv%weights:0",
        #         "I0_Attention_Model%Inception_v3%Mixed_p8_b%Branch_2%c_conv2d_2%Conv%weights:0",
        #         "I0_Attention_Model%Inception_v3%Mixed_p8_b%Branch_2%c_conv2d_3%Conv%weights:0",
        #         "I0_Attention_Model%Inception_v3%Mixed_p8_b%Branch_3%d_conv2d%Conv%weights:0",
        #         "I0_Attention_Model%Inception_v3%Mixed_p8_c%Branch_0%a_conv2d%Conv%weights:0",
        #         "I0_Attention_Model%Inception_v3%Mixed_p8_c%Branch_1%b_conv2d_1%Conv%weights:0",
        #         "I0_Attention_Model%Inception_v3%Mixed_p8_c%Branch_1%b_conv2d_2%Conv%weights:0",
        #         "I0_Attention_Model%Inception_v3%Mixed_p8_c%Branch_2%c_conv2d_1%Conv%weights:0",
        #         "I0_Attention_Model%Inception_v3%Mixed_p8_c%Branch_2%c_conv2d_2%Conv%weights:0",
        #         "I0_Attention_Model%Inception_v3%Mixed_p8_c%Branch_2%c_conv2d_3%Conv%weights:0",
        #         "I0_Attention_Model%Inception_v3%Mixed_p8_c%Branch_3%d_conv2d%Conv%weights:0",
        #         ]
        self.class_dir = ["AttentionOcr%conv_tower_fn%INCE%InceptionV3%Conv2d_1a_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Conv2d_2a_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Conv2d_2b_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Conv2d_3b_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Conv2d_4a_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5b%Branch_0%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5b%Branch_1%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5b%Branch_1%Conv2d_0b_5x5%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5b%Branch_2%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5b%Branch_2%Conv2d_0b_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5b%Branch_2%Conv2d_0c_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5b%Branch_3%Conv2d_0b_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5c%Branch_0%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5c%Branch_1%Conv2d_0b_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5c%Branch_1%Conv_1_0c_5x5%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5c%Branch_2%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5c%Branch_2%Conv2d_0b_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5c%Branch_2%Conv2d_0c_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5c%Branch_3%Conv2d_0b_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5d%Branch_0%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5d%Branch_1%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5d%Branch_1%Conv2d_0b_5x5%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5d%Branch_2%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5d%Branch_2%Conv2d_0b_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5d%Branch_2%Conv2d_0c_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5d%Branch_3%Conv2d_0b_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6a%Branch_0%Conv2d_1a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6a%Branch_1%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6a%Branch_1%Conv2d_0b_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6a%Branch_1%Conv2d_1a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6b%Branch_0%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6b%Branch_1%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6b%Branch_1%Conv2d_0b_1x7%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6b%Branch_1%Conv2d_0c_7x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6b%Branch_2%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6b%Branch_2%Conv2d_0b_7x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6b%Branch_2%Conv2d_0c_1x7%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6b%Branch_2%Conv2d_0d_7x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6b%Branch_2%Conv2d_0e_1x7%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6b%Branch_3%Conv2d_0b_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6c%Branch_0%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6c%Branch_1%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6c%Branch_1%Conv2d_0b_1x7%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6c%Branch_1%Conv2d_0c_7x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6c%Branch_2%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6c%Branch_2%Conv2d_0b_7x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6c%Branch_2%Conv2d_0c_1x7%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6c%Branch_2%Conv2d_0d_7x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6c%Branch_2%Conv2d_0e_1x7%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6c%Branch_3%Conv2d_0b_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6d%Branch_0%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6d%Branch_1%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6d%Branch_1%Conv2d_0b_1x7%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6d%Branch_1%Conv2d_0c_7x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6d%Branch_2%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6d%Branch_2%Conv2d_0b_7x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6d%Branch_2%Conv2d_0c_1x7%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6d%Branch_2%Conv2d_0d_7x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6d%Branch_2%Conv2d_0e_1x7%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6d%Branch_3%Conv2d_0b_1x1%weights:0",]
        self.font = ImageFont.truetype("simhei.ttf", 10)
        self.image_index = 0

    def draw(self,image):
        print self.layer_name
        if len(self.layer_level) == 0:
            label_txt = self.parent_dir
        if len(self.layer_level) == 1:
            label_txt = self.class_dir[self.layer_level[0]][-15:]
        if len(self.layer_level) == 2:
            label_txt = "feature_{0}".format(self.layer_level[1])
        txt_shape = self.font.getsize(label_txt)
        image_draw = ImageDraw.Draw(image)
        pos_x = self.x_bias + self.layer_shape[0] / 2 - txt_shape[0] / 2
        pos_y = self.y_bias
        image_draw.text((pos_x, pos_y), label_txt, font=self.font,fill=(0,0,0))
        if len(self.layer_level) == 2:
            img_path = os.path.join(self.parent_dir,self.class_dir[self.layer_level[0]],
                                    "heat_map_{0}.png".format(self.layer_level[1]))
            tmp_img = Image.open(img_path)
            tmp_img = tmp_img.resize((200, 70))
            img_w,img_h = tmp_img.size
            rec_w,rec_h = self.layer_shape
            pos_x = self.x_bias + rec_w / 2 - img_w / 2
            pos_y = self.y_bias + 30
            image.paste(tmp_img,(pos_x,pos_y))
        for sub_layer in self.sub_layers:
            sub_layer.draw(image)

parent_dir = "visualization_graphs/weights_heat_analyse/"
class_dir = ["AttentionOcr%conv_tower_fn%INCE%InceptionV3%Conv2d_1a_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Conv2d_2a_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Conv2d_2b_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Conv2d_3b_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Conv2d_4a_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5b%Branch_0%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5b%Branch_1%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5b%Branch_1%Conv2d_0b_5x5%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5b%Branch_2%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5b%Branch_2%Conv2d_0b_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5b%Branch_2%Conv2d_0c_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5b%Branch_3%Conv2d_0b_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5c%Branch_0%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5c%Branch_1%Conv2d_0b_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5c%Branch_1%Conv_1_0c_5x5%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5c%Branch_2%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5c%Branch_2%Conv2d_0b_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5c%Branch_2%Conv2d_0c_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5c%Branch_3%Conv2d_0b_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5d%Branch_0%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5d%Branch_1%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5d%Branch_1%Conv2d_0b_5x5%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5d%Branch_2%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5d%Branch_2%Conv2d_0b_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5d%Branch_2%Conv2d_0c_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5d%Branch_3%Conv2d_0b_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6a%Branch_0%Conv2d_1a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6a%Branch_1%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6a%Branch_1%Conv2d_0b_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6a%Branch_1%Conv2d_1a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6b%Branch_0%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6b%Branch_1%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6b%Branch_1%Conv2d_0b_1x7%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6b%Branch_1%Conv2d_0c_7x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6b%Branch_2%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6b%Branch_2%Conv2d_0b_7x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6b%Branch_2%Conv2d_0c_1x7%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6b%Branch_2%Conv2d_0d_7x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6b%Branch_2%Conv2d_0e_1x7%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6b%Branch_3%Conv2d_0b_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6c%Branch_0%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6c%Branch_1%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6c%Branch_1%Conv2d_0b_1x7%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6c%Branch_1%Conv2d_0c_7x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6c%Branch_2%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6c%Branch_2%Conv2d_0b_7x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6c%Branch_2%Conv2d_0c_1x7%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6c%Branch_2%Conv2d_0d_7x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6c%Branch_2%Conv2d_0e_1x7%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6c%Branch_3%Conv2d_0b_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6d%Branch_0%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6d%Branch_1%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6d%Branch_1%Conv2d_0b_1x7%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6d%Branch_1%Conv2d_0c_7x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6d%Branch_2%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6d%Branch_2%Conv2d_0b_7x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6d%Branch_2%Conv2d_0c_1x7%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6d%Branch_2%Conv2d_0d_7x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6d%Branch_2%Conv2d_0e_1x7%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6d%Branch_3%Conv2d_0b_1x1%weights:0",]
data_path = "/home/night/data/visual_res"
l1 = overview_layer(parent_dir,None,None,None,"H",10,0,10)
for i,class_name in enumerate(class_dir):
    l2 = overview_layer(class_dir[i],l1,i,None,"V",10,0,10)
    tmp_data = np.load(os.path.join(data_path,class_name+".npy"))
    print class_name,tmp_data.shape
    for j in range(tmp_data.shape[-1]):
        l3 = overview_layer("{0}".format(j),l2,j,(200,100),"H",10,10,10)
l1.build_shape()
l1.build_bias()
bg_img = Image.new("RGB",l1.layer_shape,color=(193,255,193))
l1.draw(bg_img)
bg_img.save("overview_of_weight_heat.png")
