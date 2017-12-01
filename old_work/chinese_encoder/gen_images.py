#coding:utf-8
from PIL import Image,ImageFont, ImageDraw,ImageOps
import os
import sys
import cv2
import numpy as np
reload(sys)
sys.setdefaultencoding('utf-8')

train_path = "/home/night/data/chinese_font/"
fonts_path = "/home/night/data/chinese_font/font/"
font_infos_path = "/home/night/data/chinese_font/font_info/"
font_true_infos_path = "/home/night/data/chinese_font/font_true_info/"
font_imgs_save_path = "/home/night/data/chinese_font/font_imgs/"
py_fonts = []
font_names = os.listdir(fonts_path)
for font_name in font_names:
    print font_name
    tmp_py_font = ImageFont.truetype(os.path.join(fonts_path, font_name),50)
    tmp_key_name = font_name.split(".")[0]
    tmp_info_file = open(os.path.join(font_infos_path,tmp_key_name+".txt"))
    tmp_true_info_file = open(os.path.join(font_true_infos_path, tmp_key_name + ".txt"),"w")
    tmp_save_path = os.path.join(font_imgs_save_path,tmp_key_name)
    if os.path.exists(tmp_save_path) == False:
        os.mkdir(tmp_save_path)
    tmp_line = tmp_info_file.readline()
    i = 0
    while tmp_line != "":
        try:
            print i
            i += 1
            # print tmp_line[:-1]
            # print font_name, tmp_line
            tmp_zi = unicode(tmp_line[:-1], "utf-8")
            # tmp_zi = u"一"
            tmp_img = Image.new("RGB", (64, 64), color=0)
            tmp_draw = ImageDraw.Draw(tmp_img)
            tmp_size = tmp_draw.textsize(tmp_zi, font=tmp_py_font)
            tmp_draw.text((32 - tmp_size[0] / 2, 32 - tmp_size[1] / 2), tmp_zi, font=tmp_py_font, fill=(255,255,255))
            if tmp_img.getbbox() is None:
                tmp_line = tmp_info_file.readline()
                continue
            tmp_img = ImageOps.invert(tmp_img)
            tmp_img.save(os.path.join(tmp_save_path, tmp_zi + ".png"))
            tmp_true_info_file.write("{0}\n".format(tmp_zi))
            # tmp_img.show()
            # print tmp_zi
            # cv2.waitKey(0)
            tmp_line = tmp_info_file.readline()
        except Exception,e:
            f_wrong = open("wrong_gen.txt", "a")
            f_wrong.write("{0}:{1}\n".format(font_name,tmp_zi))
            f_wrong.close()
            tmp_line = tmp_info_file.readline()


