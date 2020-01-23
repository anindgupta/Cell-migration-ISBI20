#import os
#import keras
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input, merge, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
#import keras.backend as K
#import numpy as np
#import pandas as pd
#from nd2_reader_new import _data_normalize
#from keras import backend as K
#from keras import optimizers
#from PIL import Image as im_read
# =============================================================================
# from keras.callbacks import (
#     EarlyStopping,
#     CSVLogger,
#     ReduceLROnPlateau,
#     LearningRateScheduler,
# )
# =============================================================================
#import tensorflow as tf
# =============================================================================
# from keras.objectives import (
#     categorical_crossentropy,
#     sparse_categorical_crossentropy,
#     binary_crossentropy,
# )
# =============================================================================
# =============================================================================
# from keras.metrics import (
#     binary_accuracy,
#     top_k_categorical_accuracy,
#     sparse_categorical_accuracy,
#     categorical_accuracy,
#     sparse_top_k_categorical_accuracy,
# )
# =============================================================================
from keras import layers
from keras.optimizers import Optimizer
#from clr_callback import CyclicLR
#from densenet_losses import f1, categorical_focal_loss, binary_focal_loss


def convolution_block(
    x, nb_channels, blk_name, dropout_rate=None, bottleneck=False, weight_decay=1e-4
):
    """
    Creates a convolution block consisting of BN-ReLU-Conv.
    Optional: bottleneck, dropout
    """

    # Bottleneck
    if bottleneck:
        bottleneckWidth = 4
        x = BatchNormalization(
            gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay)
        )(x)
        x = Activation("relu")(x)
        x = Convolution2D(
            nb_channels * bottleneckWidth,
            (1, 1),
            use_bias=False,
            kernel_regularizer=l2(weight_decay),
            name=blk_name + "_conv_1",
        )(x)
        # Dropout
        if dropout_rate:
            x = Dropout(dropout_rate)(x, training=True)

    # Standard (BN-ReLU-Conv)
    x = BatchNormalization(
        gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay)
    )(x)
    x = Activation("relu")(x)
    if bottleneck:
        l_name = blk_name + "_bconv"
    else:
        l_name = blk_name + "_conv"
    x = Convolution2D(
        nb_channels,
        (3, 3),
        padding="same",
        use_bias=False,
        kernel_regularizer=l2(weight_decay),
        name=l_name,
    )(x)

    # Dropout
    if dropout_rate:
        x = Dropout(dropout_rate)(x, training=True)

    return x


def transition_layer(
    x, nb_channels, dropout_rate=None, compression=1.0, weight_decay=1e-4
):
    """
    Creates a transition layer between dense blocks as transition, which do convolution and pooling.
    Works as downsampling.
    """

    x = BatchNormalization(
        gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay)
    )(x)
    x = Activation("relu")(x)
    x = Convolution2D(
        int(nb_channels * compression),
        (1, 1),
        padding="same",
        use_bias=False,
        kernel_regularizer=l2(weight_decay),
    )(x)

    # Adding dropout
    if dropout_rate:
        x = Dropout(dropout_rate)(x, training=True)

    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x


def dense_block_v1(
    x,
    nb_layers,
    nb_channels,
    growth_rate,
    name,
    dropout_rate=None,
    bottleneck=False,
    weight_decay=1e-4,
):
    """
    Creates a dense block and concatenates inputs
    """

    x_list = [x]
    for i in range(nb_layers):
        blk_name = name + str(i+1)
        cb = convolution_block(
            x, growth_rate, blk_name, dropout_rate, bottleneck, weight_decay
        )
        x_list.append(cb)
        x = Concatenate(axis=-1)(x_list)
        nb_channels += growth_rate
    return x, nb_channels


def DenseNet(
    input_shape=None,
    dense_blocks=3,
    dense_layers=-1,
    growth_rate=12,
    nb_classes=None,
    dropout_rate=None,
    bottleneck=False,
    compression=1.0,
    weight_decay=1e-4,
    depth=40,
    model_type=False,
    pred_activation='tanh'
):
    """
    Creating a DenseNet

    Arguments:
        input_shape  : shape of the input images. E.g. (28,28,1) for MNIST
        dense_blocks : amount of dense blocks that will be created (default: 3)
        dense_layers : number of layers in each dense block. You can also use a list for numbers of layers [2,4,3]
                       or define only 2 to add 2 layers at all dense blocks. -1 means that dense_layers will be calculated
                       by the given depth (default: -1)
        growth_rate  : number of filters to add per dense block (default: 12)
        nb_classes   : number of classes
        dropout_rate : defines the dropout rate that is accomplished after each conv layer (except the first one).
                       In the paper the authors recommend a dropout of 0.2 (default: None)
        bottleneck   : (True / False) if true it will be added in convolution block (default: False)
        compression  : reduce the number of feature-maps at transition layer. In the paper the authors recomment a compression
                       of 0.5 (default: 1.0 - will have no compression effect)
        weight_decay : weight decay of L2 regularization on weights (default: 1e-4)
        depth        : number or layers (default: 40)

    Returns:
        Model        : A Keras model instance
    """

    if nb_classes == None:
        raise Exception(
            "Please define number of classes (e.g. num_classes=10). This is required for final softmax."
        )

    if compression <= 0.0 or compression > 1.0:
        raise Exception(
            "Compression have to be a value between 0.0 and 1.0. If you set compression to 1.0 it will be turn off."
        )

    if type(dense_layers) is list:
        if len(dense_layers) != dense_blocks:
            raise AssertionError(
                "Number of dense blocks have to be same length to specified layers"
            )
    elif dense_layers == -1:
        if bottleneck:
            dense_layers = (depth - (dense_blocks + 1)) / dense_blocks // 2
        else:
            dense_layers = (depth - (dense_blocks + 1)) // dense_blocks
        dense_layers = [int(dense_layers) for _ in range(dense_blocks)]
    else:
        dense_layers = [int(dense_layers) for _ in range(dense_blocks)]

    img_input = Input(shape=input_shape)
    nb_channels = growth_rate * 2

    print("Creating DenseNet")
    print("#############################################")
    print("Dense blocks: %s" % dense_blocks)
    print("Layers per dense block: %s" % dense_layers)
    print("#############################################")
    dr_rate=0.05
    # Initial convolution layer
    x = Convolution2D(
        nb_channels,
        (3, 3),
        padding="same",
        strides=(1, 1),
        use_bias=False,
        kernel_regularizer=l2(weight_decay),
    )(img_input)
    if dropout_rate:
        x = Dropout(dr_rate)(x, training=True)
        # Building dense blocks
    
    for block in range(dense_blocks):

        # Add dense block
        name = 'conv'+"_block" + str(block+1) + "_"
        dr_rate+=0.01
        x, nb_channels = dense_block_v1(
            x,
            dense_layers[block],
            nb_channels,
            growth_rate,
            name,
            dr_rate,
            bottleneck,
            weight_decay,
        )

        if block < dense_blocks - 1:  # if it's not the last dense block
            # Add transition_block
            x = transition_layer(
                x, nb_channels, dr_rate, compression, weight_decay
            )
            nb_channels = int(nb_channels * compression)

    x = BatchNormalization(
        gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay)
    )(x)
    x = Activation("relu")(x)
    x = GlobalAveragePooling2D()(x)
    if model_type!='class':
        x = Dense(
            nb_classes,
            activation=pred_activation,
            kernel_regularizer=l2(weight_decay),
            bias_regularizer=l2(weight_decay),
        )(x)
    else:
        x = Dense(
            nb_classes,
            activation="softmax",
            kernel_regularizer=l2(weight_decay),
            bias_regularizer=l2(weight_decay),
        )(x)

    model_name = None
    if growth_rate >= 36:
        model_name = "widedense"
    else:
        model_name = "dense"

    if bottleneck:
        model_name = model_name + "b"

    if compression < 1.0:
        model_name = model_name + "c"

    return Model(img_input, x)


def densenet_like_non_temporal(
    img_row,
    img_col,
    channel,
    dense_blocks=4,
    dense_layers=-1,
    growth_rate=12,
    nb_classes=2,
    dropout_rate=True,
    bottleneck=False,
    compression=1.0,
    weight_decay=1e-4,
    depth=40,
    model_type='class',
    pred_activation='tanh'
):

    """
    Creating a DenseNet

    Arguments:
        input_shape  : shape of the input images. E.g. (28,28,1) for MNIST
        dense_blocks : amount of dense blocks that will be created (default: 3)
        dense_layers : number of layers in each dense block. You can also use a list for numbers of layers [2,4,3]
                       or define only 2 to add 2 layers at all dense blocks. -1 means that dense_layers will be calculated
                       by the given depth (default: -1)
        growth_rate  : number of filters to add per dense block (default: 12)
        nb_classes   : number of classes
        dropout_rate : defines the dropout rate that is accomplished after each conv layer (except the first one).
                       In the paper the authors recommend a dropout of 0.2 (default: None)
        bottleneck   : (True / False) if true it will be added in convolution block (default: False)
        compression  : reduce the number of feature-maps at transition layer. In the paper the authors recomment a compression
                       of 0.5 (default: 1.0 - will have no compression effect)
        weight_decay : weight decay of L2 regularization on weights (default: 1e-4)
        depth        : number or layers (default: 40)

    Returns:
        Model        : A Keras model instance
    """

    return DenseNet(
        input_shape=(img_row, img_col, channel),
        dense_blocks=dense_blocks,
        dense_layers=dense_layers,
        growth_rate=growth_rate,
        nb_classes=nb_classes,
        dropout_rate=dropout_rate,
        bottleneck=bottleneck,
        compression=compression,
        weight_decay=weight_decay,
        depth=depth,
        model_type=model_type,
        pred_activation=pred_activation
    )


if __name__ == "__main__":
    densenet_like_non_temporal
    convolution_block
    transition_layer
    dense_block_v1