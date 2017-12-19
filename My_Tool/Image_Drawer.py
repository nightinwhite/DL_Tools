from PIL import Image,ImageDraw,ImageFont

class Image_Drawer(object):
    def __init__(self,layer_name,super_layer,layer_idx,layer_shape,sub_direction,sub_interval,sub_x_bias,sub_y_bias):
        self.layer_name = layer_name
        self.super_layer = super_layer
        if self.super_layer is not None:
            tmp_list = self.super_layer.layer_level[:]
            tmp_list.append(layer_idx)
            self.layer_level = tmp_list
            self.super_layer.sub_layers.append(self)
        else:
            self.layer_level = []
        self.sub_layers = []
        self.layer_shape = None
        if len(self.sub_layers) == 0 :
            self.layer_shape = layer_shape
        self.sub_direction = sub_direction # V or H
        self.sub_interval = sub_interval
        self.sub_x_bias = sub_x_bias
        self.sub_y_bias = sub_y_bias
        self.x_bias = None
        self.y_bias = None

    def build_shape(self):
        if self.layer_shape is None:
            res_w = 0
            res_h = 0
            for sub_layer in self.sub_layers:
                tmp_w, tmp_h = sub_layer.build_shape()
                if self.sub_direction == "V":
                    if tmp_w > res_w:
                        res_w = tmp_w
                    res_h += tmp_h + self.sub_interval
                else:
                    if tmp_h > res_h:
                        res_h = tmp_h
                    res_w += tmp_w + self.sub_interval
            if self.sub_direction == "V":
                res_h -= self.sub_interval
            else:
                res_w -= self.sub_interval
            res_w += 2 * self.sub_x_bias
            res_h += 2 * self.sub_y_bias
            self.layer_shape = (res_w,res_h)
            return res_w, res_h
        else:
            return self.layer_shape

    def build_bias(self):
        if self.super_layer is None:
            self.x_bias = 0
            self.y_bias = 0
        print self.x_bias,self.y_bias,self.layer_name,self.layer_shape
        bef_bias = 0
        for sub_layer in self.sub_layers:
            sub_layer_idx = sub_layer.layer_level[-1]
            if self.sub_direction == "H":
                sub_layer.x_bias = self.x_bias + self.sub_x_bias + bef_bias
                sub_layer.y_bias = self.y_bias + self.sub_y_bias
                bef_bias += self.sub_interval + sub_layer.layer_shape[0]
            else:
                sub_layer.x_bias = self.x_bias + self.sub_x_bias
                sub_layer.y_bias = self.y_bias + self.sub_y_bias + bef_bias
                bef_bias += self.sub_interval + sub_layer.layer_shape[1]
            sub_layer.build_bias()

    def draw(self,image):
        font = ImageFont.truetype("simhei.ttf",10)
        txt_shape = font.getsize(self.layer_name)
        image_draw = ImageDraw.Draw(image)
        pos_x = self.x_bias + self.layer_shape[0]/2 - txt_shape[0]/2
        pos_y = self.y_bias
        image_draw.text((pos_x,pos_y),self.layer_name,font = font)
        print
        image_draw.rectangle(((self.x_bias,self.y_bias),(self.x_bias+self.layer_shape[0]-1,self.y_bias+self.layer_shape[1]-1)),outline="red")
        for sub_layer in self.sub_layers:
            sub_layer.draw(image)



