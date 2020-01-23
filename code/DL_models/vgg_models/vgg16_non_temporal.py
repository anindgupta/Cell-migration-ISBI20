#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 17:39:28 2019

@author: anindya
"""
from __future__ import absolute_import
from keras.layers import Input, Activation
from keras.models import Model
from keras import layers
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Dense, BatchNormalization, Flatten
from keras.applications import vgg16
from keras.layers.convolutional import Convolution2D

# from vgg_losses import categorical_focal_loss,f1
from keras.regularizers import l2


class _vgg16_imagenet_non_temporal:
    def __init__(
        self,
        img_rows,
        img_cols,
        channel,
        no_of_nodes=1,
        pred_activation="tanh",
        model_type="class",
    ):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.no_of_nodes = no_of_nodes
        self.pred_activation = pred_activation
        self.model_type = model_type

        import os

        seed_value = 10
        os.environ["PYTHONHASHSEED"] = str(seed_value)
        import random

        random.seed(seed_value)
        import numpy as np

        np.random.seed(seed_value)
        import tensorflow as tf
        from keras import backend as K

        tf.set_random_seed(seed_value)

    def _vgg_base(self):
        vgg_conv = vgg16.VGG16(
            weights="imagenet",
            include_top=False,
            input_shape=(self.img_rows, self.img_cols, self.channel),
        )
        return vgg_conv

    def _vgg16_imagenet_non_tempo(self):
        """
            this will use imagenet learnt features to fine-tune the model for
            single frame prediction
        """
        if self.model_type != "class":
            if self.no_of_nodes > 1:
                raise Exception(
                    "Please provide correct number of final nodes for a regression model"
                )

        vgg_base = _vgg16_imagenet_non_temporal(
            self.img_rows, self.img_cols, self.channel
        )._vgg_base()

        vgg_base.trainable = True
        set_trainable = False

        layer_name = "block5_conv1"
        """
            this layer is emperically found after several optimization and it is 
            recommended in literature that the fine-tunning of last layer benefits
            when data is different from imagenet dataset.
        """
        for layer in vgg_base.layers:
            if layer.name == layer_name:
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

        x = vgg_base.output
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        # x = Dense(512, activation='elu', name='fc2')(x)
        x = Dropout(0.15)(x, training=True)

        if self.model_type == "class":
            no_of_nodes = self.no_of_nodes
            final_activation = Activation("softmax")
        else:
            no_of_nodes = self.no_of_nodes
            final_activation = Activation(self.pred_activation)

        layer_name = str(self.model_type + "_predictions")
        x = Dense(no_of_nodes, name=layer_name)(x)
        predictions = final_activation(x)

        small_model = Model(inputs=vgg_base.input, outputs=predictions)

        return small_model


class _vgg16_like_non_temporal:
    def __init__(
        self,
        img_rows,
        img_cols,
        channel,
        dropout_rate=True,
        growth_rate=2,
        pred_activation="tanh",
        include_top=True,
        batch_norm=True,
        no_of_nodes=3,
        model_type="class",
    ):

        self.input_shape = (img_rows, img_cols, channel)
        self.dropout_rate = dropout_rate
        self.growth_rate = growth_rate
        self.include_top = include_top
        self.no_of_nodes = no_of_nodes
        self.batch_norm = batch_norm
        self.pred_activation = pred_activation
        self.model_type = model_type
        # self.dropout_rate_growth=dropout_rate_growth
        import os

        seed_value = 10
        os.environ["PYTHONHASHSEED"] = str(seed_value)
        import random

        random.seed(seed_value)
        import numpy as np

        np.random.seed(seed_value)
        import tensorflow as tf
        from keras import backend as K

        tf.set_random_seed(seed_value)

    def _con_block(
        _input,
        name,
        nb_filters,
        dropout_rate=None,
        dropout_rate_growth=0.05,
        batch_norm=True,
        growth_rate=2,
        weight_decay=1e-4,
    ):
        x = _input
        for num_layer in range(growth_rate):
            l_name = name + "_conv" + str(num_layer + 1)
            x = Convolution2D(
                nb_filters,
                (3, 3),
                padding="same",
                strides=(1, 1),
                use_bias=False,
                kernel_regularizer=l2(weight_decay),
                name=l_name,
            )(x)
            x = Activation("relu")(x)
            if batch_norm:
                x = BatchNormalization()(x)
            if dropout_rate:
                x = Dropout(dropout_rate_growth)(x, training=True)
        return x

    def _vgg16_like_non_tempo(self):
        """
            this will train a vgg like network for single frame prediction with 
            dropout after every layer with probablity increasing sequencitally.
        """

        nb_filters = [64, 128, 256, 512, 512]
        if self.model_type != "class":
            if self.no_of_nodes > 1:
                raise Exception(
                    "Please provide correct number of final nodes for a regression model"
                )

        img_input = Input(shape=self.input_shape, name="input")
        if self.batch_norm:
            x = BatchNormalization()(img_input)
        else:
            x = img_input

        # block-1
        x = _vgg16_like_non_temporal._con_block(
            x,
            "block_1",
            nb_filters=nb_filters[0],
            batch_norm=self.batch_norm,
            dropout_rate=self.dropout_rate,
            dropout_rate_growth=0.05,
            growth_rate=self.growth_rate,
        )

        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        # block2-block5
        block_depth = 5
        dropout_rate_growth = 0.05
        growth_rate = self.growth_rate

        for num_block in range(1, block_depth):
            name = "block_" + str(num_block + 1)
            if block_depth > len(nb_filters):
                no_filter = nb_filters[len(nb_filters) - 1]
            else:
                no_filter = nb_filters[num_block]
            if num_block >= 2:
                growth_rate = 3

            dr_rate = self.dropout_rate * num_block + dropout_rate_growth
            x = _vgg16_like_non_temporal._con_block(
                x,
                name,
                nb_filters=no_filter,
                batch_norm=self.batch_norm,
                dropout_rate=self.dropout_rate,
                dropout_rate_growth=dr_rate,
                growth_rate=growth_rate,
            )
            x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        # dense layers
        if self.include_top:
            x = Flatten()(x)
            dense_units = [512, 1024]
            dr_fc = dr_rate

            for no_of_dense in range(len(dense_units)):
                name = "fc" + str(no_of_dense + 1)
                x = Dense(dense_units[no_of_dense], name=name)(x)
                dr_fc = dr_fc + dropout_rate_growth
                x = Dropout(dr_fc)(x, training=True)
            if self.model_type == "class":
                no_of_nodes = self.no_of_nodes
                final_activation = Activation("softmax")
            else:
                no_of_nodes = self.no_of_nodes
                final_activation = Activation(self.pred_activation)
            layer_name = str(self.model_type + "fc" + str(len(dense_units) + 1))
            x = Dense(no_of_nodes, name=layer_name)(x)
            x = final_activation(x)
        else:
            x = GlobalAveragePooling2D(name="avg_pool")(x)
            if self.dropout_rate:
                dr_rate = dr_rate + dropout_rate_growth
                x = Dropout(dr_rate)(x, training=True)
            if self.model_type == "class":
                no_of_nodes = self.no_of_nodes
                final_activation = Activation("softmax")
            else:
                no_of_nodes = self.no_of_nodes
                final_activation = Activation(self.pred_activation)

            layer_name = str(self.model_type + "_predictions")
            x = Dense(no_of_nodes, name=layer_name)(x)
            x = final_activation(x)

        model = Model(inputs=img_input, outputs=x)
        return model


# model-1
def vgg16_like_non_temporal(
    img_rows,
    img_cols,
    channel,
    dropout_rate=0.05,
    growth_rate=2,
    pred_activation="tanh",
    batch_norm=True,
    include_top=True,
    no_of_nodes=3,
    model_type="class",
):

    return _vgg16_like_non_temporal(
        img_rows=img_rows,
        img_cols=img_cols,
        channel=channel,
        dropout_rate=dropout_rate,
        growth_rate=growth_rate,
        pred_activation=pred_activation,
        batch_norm=batch_norm,
        include_top=include_top,
        no_of_nodes=no_of_nodes,
        model_type=model_type,
    )._vgg16_like_non_tempo()


# model-2
def vgg16_imagenet_non_temporal(
    img_rows,
    img_cols,
    channel,
    no_of_nodes=1,
    pred_activation="tanh",
    model_type="class",
):

    return _vgg16_imagenet_non_temporal(
        img_rows=img_rows,
        img_cols=img_cols,
        channel=channel,
        no_of_nodes=no_of_nodes,
        pred_activation=pred_activation,
        model_type=model_type,
    )._vgg16_imagenet_non_tempo()


if __name__ == "__main__":
    vgg16_like_non_temporal
    vgg16_imagenet_non_temporal
