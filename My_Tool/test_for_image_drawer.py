from Image_Drawer import Image_Drawer
from PIL import Image
l0 = Image_Drawer(layer_name="first_img",super_layer=None,layer_idx=0,layer_shape=None,
                  sub_direction="H",sub_interval=10,sub_x_bias=20,sub_y_bias=20)
l0_0 = Image_Drawer(layer_name="0_0_img",super_layer=l0,layer_idx=0,layer_shape=None,
                  sub_direction="V",sub_interval=10,sub_x_bias=20,sub_y_bias=20)
l0_1 = Image_Drawer(layer_name="0_1_img",super_layer=l0,layer_idx=1,layer_shape=None,
                  sub_direction="V",sub_interval=10,sub_x_bias=20,sub_y_bias=20)
l0_2 = Image_Drawer(layer_name="0_2_img",super_layer=l0,layer_idx=2,layer_shape=None,
                  sub_direction="V",sub_interval=10,sub_x_bias=20,sub_y_bias=20)
l0_3 = Image_Drawer(layer_name="0_3_img",super_layer=l0,layer_idx=3,layer_shape=None,
                  sub_direction="V",sub_interval=10,sub_x_bias=20,sub_y_bias=20)
l0_0_0 = Image_Drawer(layer_name="0_0_0_img",super_layer=l0_0,layer_idx=0,layer_shape=(120,120),
                  sub_direction="V",sub_interval=10,sub_x_bias=20,sub_y_bias=20)
l0_0_1 = Image_Drawer(layer_name="0_0_1_img",super_layer=l0_0,layer_idx=1,layer_shape=(120,120),
                  sub_direction="V",sub_interval=10,sub_x_bias=20,sub_y_bias=20)
l0_0_2 = Image_Drawer(layer_name="0_0_2_img",super_layer=l0_0,layer_idx=2,layer_shape=(120,120),
                  sub_direction="V",sub_interval=10,sub_x_bias=20,sub_y_bias=20)
l0_0_3 = Image_Drawer(layer_name="0_0_3_img",super_layer=l0_0,layer_idx=3,layer_shape=(120,120),
                  sub_direction="V",sub_interval=10,sub_x_bias=20,sub_y_bias=20)
l0_1_0 = Image_Drawer(layer_name="0_1_0_img",super_layer=l0_1,layer_idx=0,layer_shape=(120,120),
                  sub_direction="V",sub_interval=10,sub_x_bias=20,sub_y_bias=20)
l0_1_1 = Image_Drawer(layer_name="0_1_1_img",super_layer=l0_1,layer_idx=1,layer_shape=(120,120),
                  sub_direction="V",sub_interval=10,sub_x_bias=20,sub_y_bias=20)
l0_1_2 = Image_Drawer(layer_name="0_1_2_img",super_layer=l0_1,layer_idx=2,layer_shape=(120,120),
                  sub_direction="V",sub_interval=10,sub_x_bias=20,sub_y_bias=20)
l0_2_0 = Image_Drawer(layer_name="0_2_0_img",super_layer=l0_2,layer_idx=0,layer_shape=(120,120),
                  sub_direction="V",sub_interval=10,sub_x_bias=20,sub_y_bias=20)
l0_2_1 = Image_Drawer(layer_name="0_2_1_img",super_layer=l0_2,layer_idx=1,layer_shape=(120,120),
                  sub_direction="V",sub_interval=10,sub_x_bias=20,sub_y_bias=20)
l0.build_shape()
l0.build_bias()

bg_img = Image.new("RGB",l0.layer_shape)
l0.draw(bg_img)
bg_img.show()