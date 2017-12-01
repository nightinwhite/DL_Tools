# coding:utf-8
from fontTools.ttLib import TTFont
from fontTools.unicode import Unicode
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
train_path = "/home/night/data/chinese_font/"
path = "/home/night/data/chinese_font/font/"
info_path = "/home/night/data/chinese_font/font_info/"
f_wrong = open("wrong_font.txt","w")
font_names = os.listdir(path)
for font_name in font_names:
    font_path = os.path.join(path, font_name)
    font_info_file_path = open(os.path.join(info_path, font_name.split(".")[0]+".txt"), "w")
    font = TTFont(font_path)
    exist_zi = {}
    print font_name
    for table in font["cmap"].tables:
        try:
            for item in table.cmap.items():
                unistr = item[1]
                if "uni" in unistr and len(unistr) == 7:
                    if int(unistr[3:], 16) >= int("4E00", 16) and int(unistr[3:], 16) <= int("9FCB", 16):
                        tmp_unicode = "\\u{0}".format(unistr[3:])
                        tmp_unicode = tmp_unicode.decode('unicode-escape')
                        if exist_zi.get(tmp_unicode,False) == False:
                            font_info_file_path.write("{0}\n".format(tmp_unicode))
                            exist_zi[tmp_unicode] = True

        except Exception,e:
            print e
            f_wrong.write("{0}\n".format(font_name))





