from My_Tool.Image_Drawer import *
import os
from PIL import Image
import numpy as np
class overview_layer(Image_Drawer):
    def __init__(self,layer_name,super_layer,layer_idx,layer_shape,sub_direction,sub_interval,sub_x_bias,sub_y_bias):
        Image_Drawer.__init__(self,layer_name,super_layer,layer_idx,layer_shape,sub_direction,sub_interval,sub_x_bias,sub_y_bias)
        self.parent_dir = "visualization_graphs/features_location_analyse/"
        self.class_dir = ["Conv2d_1a_3x3.npy/",
                     "Conv2d_2a_3x3.npy/",
                     "Conv2d_2b_3x3.npy/",
                     "MaxPool_3a_3x3.npy/",
                     "Conv2d_3b_1x1.npy/",
                     "Conv2d_4a_3x3.npy/",
                     "MaxPool_5a_3x3.npy/",
                     "Mixed_5b.npy/",
                     "Mixed_5c.npy/",
                     "Mixed_5d.npy/",
                     "Mixed_6a.npy/",
                     "Mixed_6b.npy/",
                     "Mixed_6c.npy/",
                     "Mixed_6d.npy/",]
        self.font = ImageFont.truetype("simhei.ttf", 10)
        self.image_index = 1

    def draw(self,image):
        print self.layer_name
        if len(self.layer_level) == 0:
            label_txt = self.parent_dir
        if len(self.layer_level) == 1:
            label_txt = self.class_dir[self.layer_level[0]]
        if len(self.layer_level) == 2:
            label_txt = "feature_{0}".format(self.layer_level[1])
        txt_shape = self.font.getsize(label_txt)
        image_draw = ImageDraw.Draw(image)
        pos_x = self.x_bias + self.layer_shape[0] / 2 - txt_shape[0] / 2
        pos_y = self.y_bias
        image_draw.text((pos_x, pos_y), label_txt, font=self.font,fill=(0,0,0))
        if len(self.layer_level) == 2:
            img_path = os.path.join(self.parent_dir,self.class_dir[self.layer_level[0]],
                                    "{0}".format(self.layer_level[1]),"{0}.png".format(self.image_index))
            tmp_img = Image.open(img_path)
            img_w,img_h = tmp_img.size
            rec_w,rec_h = self.layer_shape
            pos_x = self.x_bias + rec_w / 2 - img_w / 2
            pos_y = self.y_bias + 30
            image.paste(tmp_img,(pos_x,pos_y))
        for sub_layer in self.sub_layers:
            sub_layer.draw(image)

parent_dir = "visualization_graphs/features_location_analyse/"
# class_dir = ["I0_Conv2d_p1_3x3.npy",
#              "I0_Conv2d_p2_3x3.npy",
#              "I0_Conv2d_p3_3x3.npy",
#              "I0_MaxPool_p4_3x3.npy",
#              "I0_Conv2d_p5_1x1.npy",
#              "I0_Conv2d_p6_3x3.npy",
#              "I0_MaxPool_p7_3x3.npy",
#              "I0_Mixed_p8_a.npy",
#              "I0_Mixed_p8_b.npy",
#              "I0_Mixed_p8_c.npy", ]
class_dir = ["Conv2d_1a_3x3.npy",
                     "Conv2d_2a_3x3.npy",
                     "Conv2d_2b_3x3.npy",
                     "MaxPool_3a_3x3.npy",
                     "Conv2d_3b_1x1.npy",
                     "Conv2d_4a_3x3.npy",
                     "MaxPool_5a_3x3.npy",
                     "Mixed_5b.npy",
                     "Mixed_5c.npy",
                     "Mixed_5d.npy",
                     "Mixed_6a.npy",
                     "Mixed_6b.npy",
                     "Mixed_6c.npy",
                     "Mixed_6d.npy",
                     ]
data_path = "/home/night/data/key_points/"
l1 = overview_layer(parent_dir,None,None,None,"H",10,0,10)
for i,class_name in enumerate(class_dir):
    l2 = overview_layer(class_dir[i],l1,i,None,"V",10,0,10)
    tmp_data = np.load(os.path.join(data_path,class_name))
    for j in range(tmp_data.shape[-1]):
        l3 = overview_layer("{0}".format(j),l2,j,(100,100),"H",10,10,10)
l1.build_shape()
l1.build_bias()
bg_img = Image.new("RGB",l1.layer_shape,color=(193,255,193))
l1.draw(bg_img)
bg_img.save("overview_of_feature_location{0}.png".format(l1.image_index))
