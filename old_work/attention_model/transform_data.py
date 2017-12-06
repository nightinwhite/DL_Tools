from My_Data.Data_reader_config import Data_Reader_Config
from My_Data.Data_to_tf import Data_To_TF
import os
from PIL import Image
import numpy as np


file_path = "/home/night/data/chinese1/val/"
tf_file = "tfrecords/chinese1_val.record"
file_names = os.listdir(file_path)
file_names.sort(key= lambda a: int(a.split("_")[0]))
print file_names
# file_names = file_names[:10000]
read_data = []

for f_name in file_names:
    tmp_path = os.path.join(file_path, f_name)
    tmp_label = f_name.split("_")[1]
    tmp_label = tmp_label.split(".")[0]
    print tmp_path, tmp_label
    read_data.append([tmp_path, unicode(tmp_label,"utf-8")])

def parse_data_fuc(tmp_path):
    img = Image.open(tmp_path)
    img = np.asarray(img, np.uint8)
    img = img.tobytes()
    return img

# char_to_label = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,
#                  '6':6,'7':7,'8':8,'9':9,'a':10,'b':11,
#                  'c':12,'d':13,'e':14,'f':15,'g':16,
#                  'h':17,'i':18,'j':19,'k':20,'l':21,
#                  'm':22,'n':23,'o':24,'p':25,'q':26,
#                  'r':27,'s':28,'t':29,'u':30,'v':31,
#                  'w':32,'x':33,'y':34,'z':35,'A':36,
#                  'B':37,'C':38,'D':39,'E':40,'F':41,
#                  'G':42,'H':43,'I':44,'J':45,'K':46,
#                  'L':47,'M':48,'N':49,'O':50,'P':51,
#                  'Q':52,'R':53,'S':54,'T':55,'U':56,
#                  'V':57,'W':58,'X':59,'Y':60,'Z':61,
#                  '#':62}

f = open("chinese1")
char_to_label = {}
tmp_line = f.readline()
while tmp_line != "":
    tmp_char = tmp_line.split(" ")[0]
    tmp_label = tmp_line.split(" ")[1][:-1]
    char_to_label[unicode(tmp_char,"utf-8")] = int(tmp_label)
    tmp_line = f.readline()

data_read_config = Data_Reader_Config(read_data=read_data,
                                          parse_data_fuc=parse_data_fuc,
                                          is_shuffle=False,
                                          with_label=True,
                                          label_length=4,
                                          char_to_label_dict=char_to_label,
                                          fill_label=3755,
                                          thread_num=4
                                          )
data_to_tf = Data_To_TF(data_read_config,tf_file)
data_to_tf.generate()
