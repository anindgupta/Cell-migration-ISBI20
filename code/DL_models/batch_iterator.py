#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 16:00:50 2019

@author: anindya
"""

import tifffile
from sklearn.utils import shuffle
import pandas as pd
import numpy as np

class _data_normalize:
    def _list_touint8(list_data):
        modified_list = []
        for img in range(len(list_data)):
            modified_list.append(_data_normalize._touint8(list_data[img]))
        return modified_list

    def _touint8(img):
        imgout = np.float32(img)
        imgout = ((imgout - imgout.min()) / (imgout.max() - imgout.min())) * 255.0
        return np.uint8(imgout)

    def _normalize(img):
        imgout = np.float32(img)
        # imgout=(imgout - imgout.min()) / (imgout.max() - imgout.min())
        imgout = imgout / imgout.max()
        return imgout

    def _channel_touint8(img):
        imgout = np.float32(img)
        imgs_normalized = np.empty((img.shape), dtype="uint8")
        if len(imgout.shape) >= 4:
            for i in range(img.shape[0]):
                for ii in range(img.shape[3]):
                    imgs_normalized[i, :, :, ii] = _data_normalize._touint8(
                        imgout[i, :, :, ii]
                    )  
        else:
            for i in range(img.shape[2]):
                imgs_normalized[:, :, i] = _data_normalize._touint8(
                    imgout[:, :, i]
                )  

        return imgs_normalized

    def _channelnormalize(img):
        imgout = img
        imgs_normalized = np.empty((img.shape), dtype="float32")
        if len(imgout.shape) >= 4:
            for i in range(img.shape[0]):
                for ii in range(img.shape[3]):
                    imgs_normalized[i, :, :, ii] = _data_normalize._normalize(
                        imgout[i, :, :, ii]
                    ) 
        else:
            for i in range(img.shape[2]):
                imgs_normalized[:, :, i] = _data_normalize._normalize(
                    imgout[:, :, i]
                )  

        return imgs_normalized


def batch_iterator_non_temporal(
    fold_file,
    img_rows,
    img_cols,
    channel,
    batch,
    model_name,
    preprocess_input,
    regression=False,
):

    # reading fold files:
    data = shuffle(pd.read_excel(fold_file))
    data = data.drop(data[(data.Class_label == 3)].index).reset_index(drop=True)
    image_path = list(data.pop("fileName"))

    im_no = list(data.pop("Patch_ID"))
    if regression:
        class_label = list(data.pop("Reg_label"))
    else:
        class_label = list(data.pop("Class_label"))
        class_label=pd.get_dummies(class_label).values


    # batch=32 #number of original samples : 32 from the fold file
    idx = 0

    while 1:

        tmp = np.zeros((batch, img_rows, img_cols, channel))
        if regression:
            y = np.zeros([batch, 1, 1], dtype="float32").reshape([batch, 1])
        else:
            y = np.zeros([batch, 1, 2], dtype="int32").reshape([batch, 2])

        for i in range(batch):

            ii = idx * batch + (i)

            ii = ii % (len(im_no) - 10)  # to track back to the starting image pointer
            if len(tifffile.imread(image_path[ii]).shape) < 3:
                print("it is a single channel image data")
                image_to_read = tifffile.imread(image_path[ii])
                image_to_read = _data_normalize._touint8(image_to_read)[np.newaxis, ...]
            else:
                image_to_read = tifffile.imread(image_path[ii])
                
                image_to_read =image_to_read

            y[i, :] = class_label[ii]
            tmp[i, :, :, :] = image_to_read
        tmp_processed = preprocess_input(tmp)
        idx = idx + 1
        yield tmp_processed, y
