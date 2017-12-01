#coding:utf-8
import Queue
import threading
import numpy as np
import os
import cv2
import sys
reload(sys)
sys.setdefaultencoding('utf8')

class Image_Reader(object):
    def __init__(self, min_of_queue, max_of_queue,thread_num,font_info_path,font_img_path):

        self.min_of_queue = min_of_queue
        self.max_of_queue = max_of_queue
        self.mutex = threading.Lock()
        self.queue = Queue.Queue(self.max_of_queue)
        self.fonts_in_char = {}
        font_info_path = font_info_path
        font_info_names = os.listdir(font_info_path)
        for f_name in font_info_names:
            tmp_file = open(os.path.join(font_info_path,f_name),"r")
            tmp_line = tmp_file.readline()
            while tmp_line != "":
                tmp_line = unicode(tmp_line, "utf-8")
                tmp_key = tmp_line[:-1]
                tmp_font_list = self.fonts_in_char.get(tmp_key,[])
                tmp_font_list.append(f_name.split(".")[0])
                self.fonts_in_char[tmp_key] = tmp_font_list
                tmp_line = tmp_file.readline()
        self.font_img_path = font_img_path
        self.chars = self.fonts_in_char.keys()
        self.threads = []
        for t in xrange(thread_num):
            tmp_thread = threading.Thread(target=self.put_data,args=(t,))
            tmp_thread.setDaemon(True)
            tmp_thread.start()
            self.threads.append(tmp_thread)

    def get_single_image(self,path):
        # print path
        img = cv2.imread(path)
        # img = cv2.resize(img,(64,64),interpolation=cv2.INTER_AREA)
        # img = img/255.
        return img

    def put_data(self,t):
        while True:
            if self.queue.full() == False:
                tmp_key = np.random.choice(self.chars)
                tmp_font_list = self.fonts_in_char[tmp_key]
                while len(tmp_font_list) < 2:
                    tmp_key = np.random.choice(self.chars)
                    tmp_font_list = self.fonts_in_char[tmp_key]
                # print self.fonts_in_char
                tmp_font = np.random.choice(tmp_font_list)
                tmp_font_same_class = np.random.choice(tmp_font_list)
                while tmp_font_same_class == tmp_font:
                    tmp_font_same_class = np.random.choice(tmp_font_list)
                # tmp_img_path = os.path.join(self.font_img_path,tmp_font,tmp_key+".png")
                tmp_img_path = self.font_img_path + tmp_font + "/" + tmp_key+".png"

                tmp_path = self.font_img_path+tmp_font
                tmp_imgs_same_style_path = os.listdir(tmp_path)

                tmp_img_same_style_path = np.random.choice(tmp_imgs_same_style_path)
                while tmp_img_same_style_path == tmp_key+".png":
                    tmp_img_same_style_path = np.random.choice(tmp_imgs_same_style_path)
                tmp_path = self.font_img_path+tmp_font+"/"+tmp_img_same_style_path
                tmp_img_same_style_path = tmp_path
                tmp_path = self.font_img_path+tmp_font_same_class+"/"+tmp_key+".png"
                tmp_img_same_class_path = tmp_path
                tmp_img = self.get_single_image(tmp_img_path)
                if tmp_img is None:
                    continue
                tmp_img_same_style = self.get_single_image(tmp_img_same_style_path)
                if tmp_img_same_style is None:
                    continue
                tmp_img_same_class = self.get_single_image(tmp_img_same_class_path)
                if tmp_img_same_class is None:
                    continue
                tmp_res = [tmp_img, tmp_img_same_class, tmp_img_same_style]
                tmp_res = np.asarray(tmp_res)
                self.queue.put(tmp_res)

    def get_batch_data(self,batch_size):
        while self.queue.qsize() < self.min_of_queue:
            # print self.queue.qsize()
            pass
        res = []
        for i in range(batch_size):
            res.append(self.queue.get())
        res = np.asarray(res)
        return res
