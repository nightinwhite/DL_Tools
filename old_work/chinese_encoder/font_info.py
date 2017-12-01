#coding:utf-8
from fontTools.ttLib import TTFont
from fontTools.unicode import Unicode
import os
path = "font/"
font_name = os.listdir(path)
for f_name in font_name:
    f_path = os.path.join(path,f_name)
    print f_path
    tmp_font = TTFont(f_path)
    f = open("font_info/{0}_ttf.txt".format(f_name.split(".")[0]),"w")
    for table in tmp_font["cmap"].tables:
        for item in table.cmap.items():
            tmp_unicode = item[1]
            if "uni" in tmp_unicode and len(tmp_unicode) == 7:
                print tmp_unicode
                tmp_unicode = tmp_unicode[3:]
                tmp_res = "\\u{0}".format(tmp_unicode)
                char = tmp_res.decode("unicode_escape")
                print char
                number = tmp_unicode
                f.write("{0} {1}\n".format(char.encode("gbk"), number))