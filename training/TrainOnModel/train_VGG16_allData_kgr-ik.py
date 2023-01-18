import os
from collections import deque
from pathlib import Path
import re
import tensorflow as tf
import numpy as np
import pandas as pd


def GetImageFiles(dirPath):
    return list(Path(dirPath).rglob('*.[pP][nN][gG]'))

def train_VGG16(imgPath):
    #if not (os.path.exists(out_dir)):
       # os.mkdir(out_dir)

    train_ds, test_ds = load_samples(imgPath)
    print(train_ds.element_spec)
    print(test_ds.element_spec)


def load_samples(dirPath):
    train_data_path = os.path.join(dirPath, 'train')
    train_data = []
    train_labels = []
    list_train_data = os.listdir(train_data_path)
    for file_name in list_train_data:
        tmp_df = pd.read_csv(os.path.join(train_data_path, file_name))
        labels = tmp_df.pop('classname')
        train = tf.data.Dataset.from_tensor_slices((tmp_df.values, labels.values))
        train_ds = tf.data.Dataset.zip(train_ds, train)
        label_list = list(tmp_df['label'])
        img_list = list(tmp_df['filename'])
        train_data.append(img_list)
        train_labels.append(label_list)

    train_data = np.asarray(train_data)
    train_labels = np.asarray(train_labels)

    test_data_path = os.path.join(dirPath, 'test')
    test_data = []
    test_labels = []
    list_test_data = os.listdir(test_data_path)
    for file_name in list_test_data:
        tmp_df = pd.read_csv(os.path.join(test_data_path, file_name))
        labels = tmp_df.pop('classname')
        test = tf.data.Dataset.from_tensor_slices((tmp_df.values, labels.values))
        test_ds = tf.data.Dataset.zip(test_ds, test)

    return train_ds, test_ds

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

imgPath = 'D:\\MA_Inna\\data\\csv_files\\x_view'
train_VGG16(imgPath=imgPath)