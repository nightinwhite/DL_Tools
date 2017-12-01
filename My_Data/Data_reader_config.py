#coding:utf-8
class Data_Reader_Config(object):
    def __init__(self,read_data,parse_data_fuc,is_shuffle,with_label,
                 label_length = None,char_to_label_dict=None,fill_label=None,
                 thread_num = 4,):
        self.read_data = read_data
        self.with_label = with_label
        self.is_shuffle = is_shuffle
        self.label_length = label_length
        self.char_to_label_dict = char_to_label_dict
        self.thread_num = thread_num
        self.fill_label = fill_label
        self.parse_data_fuc = parse_data_fuc #处理到rawdata,resize,不除255
