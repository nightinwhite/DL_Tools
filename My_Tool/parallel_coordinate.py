from PIL import ImageDraw,ImageFont,Image
import pygame
import numpy as np
class Gen_parallel_coordinate(object):
    def __init__(self,data,
                 axis_len = 40,axis_height = 200,axis_bias = 25,
                 axis_width = 3,line_width = 1):
        self.data = data
        self.axis_len = axis_len
        self.axis_bias = axis_bias
        self.axis_height = axis_height
        self.axis_width = axis_width
        self.line_width = line_width
        self.boundary_multi = 2
        self.line_pos = []
        self.img_width, self.img_height,self.img_x_axis_height = self.get_shape()
        self.bg_img = np.full((self.img_height, self.img_width, 4), 255, np.uint8)
        self.bg_img = pygame.image.frombuffer(self.bg_img.tostring(), (self.img_width, self.img_height), "RGBA")
        self.draw_img = np.full((self.img_height, self.img_width, 4), 0, np.uint8)
        self.draw_img = pygame.image.frombuffer(self.draw_img.tostring(), (self.img_width, self.img_height), "RGBA")
        self.font_render = pygame.font.Font("font/simhei.ttf",12)
        self.draw_bg_img()


    def get_single_struct_data_shape(self,single_data):
        single_data_shape = single_data.shape
        single_width = 0
        single_height = 0
        for i in range(len(single_data_shape)):
            index = len(single_data_shape) - i - 1
            if i == 0:
                single_width = (single_data_shape[index] + 1) * self.axis_len
                single_height = self.axis_height + 2 * self.axis_bias
            else:
                single_width = single_width * single_data_shape[index]
                single_height += 2 * self.axis_bias
        return single_width,single_height

    def get_shape(self):
        one_epoch_data = self.data[0]
        img_width = 0
        img_height = 0
        img_x_axis_height = 0
        for struct_data in one_epoch_data:
            tmp_width, tmp_height = self.get_single_struct_data_shape(struct_data)
            if tmp_height > img_height:
                img_height = tmp_height
                img_x_axis_height = self.axis_bias*len(struct_data.shape) + self.axis_height
            img_width += tmp_width
        return img_width, img_height,img_x_axis_height

    def draw_single_struct_data_bg_img(self,single_data,x_bias):
        y_bias = self.img_x_axis_height - \
                 (self.axis_bias*len(single_data.shape) + self.axis_height)

        def draw_single(single_data,x_bias,y_bias):
            single_width, single_height = self.get_single_struct_data_shape(single_data)
            if len(single_data.shape) == 1:
                pygame.draw.rect(self.draw_img,(0,0,0,20),
                                 pygame.Rect(x_bias,y_bias,single_width,single_height),self.axis_width)
                for i in range(len(single_data)):
                    tmp_x = x_bias + (i+1)*self.axis_len
                    tmp_y = y_bias + self.axis_bias + self.axis_height
                    pygame.draw.line(self.draw_img,(0,0,0,255),
                                     (tmp_x,tmp_y),(tmp_x,tmp_y - self.axis_height),self.axis_width)
                    self.line_pos.append([tmp_x, tmp_y])
                return single_width, single_height
            else:
                pygame.draw.rect(self.draw_img, (0, 0, 0, 20),
                                 pygame.Rect(x_bias, y_bias, single_width, single_height),self.axis_width)
                tmp_x_bias = x_bias
                tmp_y_bias = y_bias + self.axis_bias
                for i,tmp_data in enumerate(single_data):
                    tmp_w,tmp_h = draw_single(tmp_data,tmp_x_bias,tmp_y_bias)
                    tmp_x_bias += tmp_w
                return single_width, single_height
        return draw_single(single_data,x_bias,y_bias)

    def draw_bg_img(self):
        one_epoch_data = self.data[0]
        tmp_x_bias = 0
        for struct_data in one_epoch_data:
            tmp_w, tmp_h = self.draw_single_struct_data_bg_img(struct_data, tmp_x_bias,)
            tmp_x_bias += tmp_w

    def draw_data_line(self):
        data_in_lines = []
        for one_epoch_data in self.data:
            tmp_v_list = []
            for single_struct_data in one_epoch_data:
                tmp_shape = single_struct_data.shape
                flatten_size = 1
                for s in tmp_shape:
                    flatten_size *= s
                tmp_data_list = np.reshape(single_struct_data,flatten_size)
                for tmp_v in tmp_data_list:
                    tmp_v_list.append(tmp_v)
            data_in_lines.append(tmp_v_list)
        data_in_lines = np.asarray(data_in_lines)
        epochs,lines_num = data_in_lines.shape
        for i in range(lines_num - 1):
            data_in_single_line_b = data_in_lines[:,i]
            data_in_single_line_e = data_in_lines[:,i+1]
            b_max = float((int(np.max(data_in_single_line_b))/self.boundary_multi+1)*self.boundary_multi)
            b_min = float((int(np.min(data_in_single_line_b))/self.boundary_multi-1)*self.boundary_multi)
            e_max = float((int(np.max(data_in_single_line_e))/self.boundary_multi+1)*self.boundary_multi)
            e_min = float((int(np.min(data_in_single_line_e))/self.boundary_multi-1)*self.boundary_multi)
            for e in range(epochs):
                b_x = self.line_pos[i][0]
                b_y = self.line_pos[i][1]
                b_y = b_y - int((data_in_single_line_b[e] - b_min)/(b_max - b_min)*self.axis_height)

                e_x = self.line_pos[i+1][0]
                e_y = self.line_pos[i+1][1]
                e_y = e_y - int((data_in_single_line_e[e] - e_min) / (e_max - e_min) * self.axis_height)
                alpha = int((e+1.)**3/(epochs**3)*255)
                pygame.draw.line(self.draw_img,(0,0,0,alpha),(b_x, b_y),(e_x, e_y),2)

                b_min_b_surface = self.font_render.render("{0}".format(b_min), True, (0, 0, 0, 255))
                tmp_w, tmp_h = b_min_b_surface.get_size()
                self.draw_img.blit(b_min_b_surface, (self.line_pos[i][0] - tmp_w / 2, self.line_pos[i][1] + tmp_h / 2))
                b_max_b_surface = self.font_render.render("{0}".format(b_max), True, (0, 0, 0, 255))
                tmp_w, tmp_h = b_max_b_surface.get_size()
                self.draw_img.blit(b_max_b_surface, (
                self.line_pos[i][0] - tmp_w / 2, self.line_pos[i][1] - self.axis_height - tmp_h / 2))

                e_min_b_surface = self.font_render.render("{0}".format(e_min), True, (0, 0, 0, 255))
                tmp_w, tmp_h = e_min_b_surface.get_size()
                self.draw_img.blit(e_min_b_surface, (self.line_pos[i+1][0] - tmp_w / 2, self.line_pos[i+1][1] + tmp_h / 2))
                e_max_b_surface = self.font_render.render("{0}".format(e_max), True, (0, 0, 0, 255))
                tmp_w, tmp_h = e_max_b_surface.get_size()
                self.draw_img.blit(e_max_b_surface, (
                self.line_pos[i+1][0] - tmp_w / 2, self.line_pos[i+1][1] - self.axis_height - tmp_h / 2))

    def gen_image(self):
        self.draw_data_line()
        return self.bg_img,self.draw_img






