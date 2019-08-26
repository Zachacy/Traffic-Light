from __future__ import print_function
import xml.etree.ElementTree as ET
import numpy as np
import torch
import cv2,sys
from ctypes import *

image_size = [200,75]
CLASSES = ['Red','Yellow','Green', 'Right' , 'Left', 'Straight']#
#DATA_ROOT = "/home/rvl/Dataset/traffic_light_data/Traffic_light_classify/"
DATA_ROOT = "/home/rvl/Dataset/New_TL/"
def read_data_index(index_name) :
    data_index_path = DATA_ROOT + index_name
    data_indexs = open(data_index_path).readlines()
    data = []
    for data_index in data_indexs:
        data.append(data_index[:-5])
    return data

def batch_train_data(data,init_index,bath_num,CLASSES) :
    batch_data_indexs = [data[i] for i in range(init_index, init_index + bath_num)]
    batch_data = []
    gt_data = []
    for batch_data_index in batch_data_indexs:
        cls_id = np.zeros(len(CLASSES))
        img = cv2.imread(DATA_ROOT + 'JPEGImages/' + batch_data_index + '.jpg')
        img = cv2.resize(img, (image_size[0], image_size[1]), interpolation=cv2.INTER_CUBIC)
        batch_data.append(img)
        labels = open(DATA_ROOT + 'Annotations/' + batch_data_index + '.xml')
        tree = ET.parse(labels)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in CLASSES :#or int(difficult) == 1:
                continue
            cls_id += np.eye(len(CLASSES))[CLASSES.index(cls)]
        gt_data.append(cls_id)
    return torch.tensor(batch_data), torch.tensor(gt_data)
    #return np.array(batch_data),np.array(gt_data)
if __name__ == '__main__' :	
    data_index = sys.argv[1]
    data = read_data_index(data_index)
    batch_xs, batch_ys = batch_train_data(data=data,init_index=0,bath_num=10,CLASSES=CLASSES)
    print(np.array(data).shape)
    print(batch_xs.size())
    print(batch_ys)