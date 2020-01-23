#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 18:03:10 2019

@author: anindya
"""

import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
from DL_models.vgg_models.vgg_model_training import perfrom_training as vgg_training
from DL_models.densenet_models.densenet_model_training import perfrom_training as densenet_training
from DL_models.resnet_models.resnet_model_training import perfrom_training as resnet_training

models_to_train= [#"vgg16_imagenet_non_temporal", 
                  "vgg16_like_non_temporal",
                  #"densenet_imagenet_non_temporal", 
                  "densenet_like_non_temporal",
                  #"resnet_imagenet_non_temporal",
                  "resnet_like_non_temporal"
                 ]


img_row=227
img_col=227
channel=3
batch=16
epoch=8
final_layer_no_of_nodes=1
final_ac_regression='sigmoid'
regression_flag=True
dropout_rate_flag=True
batch_norm_flag=True
early_stopping=True
five_fold_flag=False

for mod_name in range(len(models_to_train)):
    running_model=models_to_train[mod_name]
    if running_model.split('_')[0]=='vgg16':
        vgg_training(
            models=running_model,
            img_row=img_row,
            img_col=img_col,
            channel=channel,
            batch=batch,
            epoch=epoch,
            final_layer_no_of_nodes=final_layer_no_of_nodes,
            final_ac_regression=final_ac_regression,
            regression_flag=regression_flag,
            dropout_rate_flag=dropout_rate_flag,
            batch_norm_flag=batch_norm_flag,
            early_stopping=early_stopping,
            five_fold_flag=five_fold_flag,
        )
    elif running_model.split('_')[0]=='densenet':
        densenet_training(
            models=running_model,
            img_row=img_row,
            img_col=img_col,
            channel=channel,
            batch=batch,
            epoch=epoch,
            final_layer_no_of_nodes=final_layer_no_of_nodes,
            final_ac_regression=final_ac_regression,
            regression_flag=regression_flag,
            dropout_rate_flag=dropout_rate_flag,
            batch_norm_flag=batch_norm_flag,
            early_stopping=early_stopping,
            five_fold_flag=five_fold_flag,
        )
    else:
        resnet_training(
            models=running_model,
            img_row=img_row,
            img_col=img_col,
            channel=channel,
            batch=batch,
            epoch=epoch,
            final_layer_no_of_nodes=final_layer_no_of_nodes,
            final_ac_regression=final_ac_regression,
            regression_flag=regression_flag,
            dropout_rate_flag=dropout_rate_flag,
            batch_norm_flag=batch_norm_flag,
            early_stopping=early_stopping,
            five_fold_flag=five_fold_flag,
    )
    