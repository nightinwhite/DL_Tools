from My_Data.Data_to_tf import Data_To_TF
from My_Data.Data_reader_config import Data_Reader_Config
import os
from PIL import Image
import numpy as np

file_path = "/home/night/data/fsns_validation/"
save_path = "/home/night/data/fsns_tf_validation/"
subfile_paths = os.listdir(file_path)
char_to_label = {'\x87': 83, '\x8b': 87, '\x93': 89, '\x9b': 92,
                 ' ': 0, '\xa7': 96, '(': 5, '\xab': 100, ',': 8,
                 '\xaf': 102, '0': 12, '4': 16, '8': 20, '\xbb': 105,
                 '<': 23, '\xbf': 107, '\xc3': 108, 'D': 29, 'H': 33,
                 'L': 37, 'P': 41, 'T': 45, 'X': 49, '\\': 52, 'd': 58,
                 'h': 62, 'l': 66, 'p': 70, 't': 74, 'x': 78, '\x80': 81,
                 '\x88': 84, '\x94': 90, '\xa0': 93, '\xa4': 95, "'": 4,
                 '\xa8': 97, '+': 7, '/': 11, '3': 15, '\xb4': 103,
                 '7': 19, ';': 22, '\xbc': 106, '?': 25, 'C': 28,
                 'G': 32, 'K': 36, 'O': 40, 'S': 44, 'W': 48, '_': 54,
                 'c': 57, 'g': 61, 'k': 65, 'o': 69, 's': 73, 'w': 77,
                 '\x89': 85, '\x99': 91, '"': 2, '&': 3, '\xa9': 98,
                 '.': 10, '2': 14, '6': 18, '\xb9': 104, 'B': 27,
                 '\xc5': 109, 'F': 31, 'J': 35, 'N': 39, 'R': 43,
                 'V': 47, 'Z': 51, 'b': 56, 'f': 60, 'j': 64, 'n': 68,
                 'r': 72, 'v': 76, 'z': 80, '\x82': 82, '\x8a': 86,
                 '\x8e': 88, '!': 1, '\xa2': 94, ')': 6, '\xaa': 99,
                 '-': 9, '\xae': 101, '1': 13, '5': 17, '9': 21,
                 '=': 24, 'A': 26, 'E': 30, 'I': 34, 'M': 38, 'Q': 42,
                 'U': 46, 'Y': 50, ']': 53, 'a': 55, '\xe2': 110,
                 'e': 59, 'i': 63, 'm': 67, 'q': 71, 'u': 75, 'y': 79,'#':111}
for subpath in subfile_paths:
    tf_file = os.path.join(save_path,subpath+".record")
    print tf_file
    tmp_path = os.path.join(file_path,subpath)
    tmp_file = open(os.path.join(tmp_path, "label.txt"), "r")
    tmp_line = tmp_file.readline()
    read_data = []
    while tmp_line != "":
        tmp_index = tmp_line.find(" ")
        tmp_img_name = tmp_line[:tmp_index]
        tmp_chars = tmp_line[tmp_index + 1:-1]
        tmp_img_name = os.path.join(tmp_path, tmp_img_name + ".png")
        read_data.append([tmp_img_name, tmp_chars])
        tmp_line = tmp_file.readline()


    def parse_data_fuc(tmp_path):
        img = Image.open(tmp_path)
        img = np.asarray(img,np.uint8)
        img = img.tobytes()
        return img

    data_read_config = Data_Reader_Config(read_data=read_data,
                                          parse_data_fuc=parse_data_fuc,
                                          is_shuffle=False,
                                          with_label=True,
                                          label_length=40,
                                          char_to_label_dict=char_to_label,
                                          fill_label=111,
                                          thread_num=4
                                          )
    data_to_tf = Data_To_TF(data_read_config,tf_file)
    data_to_tf.generate()
