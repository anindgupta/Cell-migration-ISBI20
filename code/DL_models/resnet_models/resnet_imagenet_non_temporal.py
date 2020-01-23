from __future__ import absolute_import
from keras.layers import Input, Activation
from keras.models import Model
from keras import layers
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Dense, BatchNormalization, Flatten
from keras.layers.convolutional import Convolution2D

# from vgg_losses import categorical_focal_loss,f1
from keras.regularizers import l2
from keras.applications import resnet50


class _resnet_imagenet_non_temporal:
    def __init__(
        self,
        img_rows,
        img_cols,
        channel,
        no_of_nodes=1,
        pred_activation="tanh",
        model_type="class",
        dropout_rate=True,
    ):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.no_of_nodes = no_of_nodes
        self.pred_activation = pred_activation
        self.model_type = model_type
        self.dropout_rate=dropout_rate
        if self.dropout_rate:
            self.dr_rate = 0.05
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

    def _resnet_base(self):
        resnet_conv = resnet50.ResNet50(
            weights="imagenet",pooling='avg',
            include_top=False,
            input_shape=(self.img_rows, self.img_cols, self.channel),
        )
        return resnet_conv

    def _resnet_imagenet_non_tempo(self):
        """
            this will use imagenet learnt features to fine-tune the model for
            single frame prediction
        """
        if self.model_type != "class":
            if self.no_of_nodes > 1:
                raise Exception(
                    "Please provide correct number of final nodes for a regression model"
                )

        resnet_base = _resnet_imagenet_non_temporal(
            self.img_rows, self.img_cols, self.channel
        )._resnet_base()

        resnet_base.trainable = True
        set_trainable = False

        layer_name = 'res3a_branch2a'
        """
            this layer is emperically found after several optimization and it is 
            recommended in literature that the fine-tunning of last layer benefits
            when data is different from imagenet dataset.
        """
        for layer in resnet_base.layers:
            if layer.name == layer_name:
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

        x = resnet_base.output
        #x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        # x = Dense(512, activation='elu', name='fc2')(x)
        if self.dropout_rate:
            x = Dropout(self.dr_rate)(x, training=True)

        if self.model_type == "class":
            no_of_nodes = self.no_of_nodes
            final_activation = Activation("softmax")
        else:
            no_of_nodes = self.no_of_nodes
            final_activation = Activation(self.pred_activation)

        layer_name = str(self.model_type + "_predictions")
        x = Dense(no_of_nodes, name=layer_name)(x)
        predictions = final_activation(x)

        small_model = Model(inputs=resnet_base.input, outputs=predictions)

        return small_model


def resnet_imagenet_non_temporal(
    img_rows,
    img_cols,
    channel,
    no_of_nodes=1,
    pred_activation="tanh",
    model_type="class",
    dropout_rate=True,
):
    return _resnet_imagenet_non_temporal(
        img_rows=img_rows,
        img_cols=img_cols,
        channel=channel,
        no_of_nodes=no_of_nodes,
        pred_activation=pred_activation,
        model_type=model_type,
        dropout_rate=dropout_rate,
    )._resnet_imagenet_non_tempo()


if __name__ == "__main__":
    resnet_imagenet_non_temporal
    _resnet_imagenet_non_temporal

