from __future__ import division

import six
from keras.models import Model
from keras import layers
from keras.layers import (
    Input,
    Activation,
    Dropout,
    Dense
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K


def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    dr_rate=conv_params['dr_rate']
    name=conv_params['name']
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer,name=name)(input)
        if dr_rate:
            conv = Dropout(dr_rate)(conv, training=True)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    dr_rate=conv_params['dr_rate']
    name=conv_params['name']
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        conv=Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer,name=name)(activation)
        if dr_rate:
            conv = Dropout(dr_rate)(conv, training=True)
        return conv
    return f


def _shortcut(input, residual,dr_rate):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)
        if dr_rate:
            shortcut = Dropout(dr_rate)(shortcut, training=True)
    return add([shortcut, residual])


def _residual_block(block_function,dropout_rate, filters,name, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            as_name=name+str('_conv_')+str(i+1)
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(filters=filters,dr_rate=dropout_rate,name=as_name,init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def basic_block(filters,dr_rate,name,init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):
        ids=1
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            asname=name+'_'+str(ids)
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4),name=asname)(input)
            if dr_rate:
                conv1 = Dropout(dr_rate)(conv1, training=True)
        else:
            asname=name+'_'+str(ids)
            conv1 = _bn_relu_conv(filters=filters,name=asname, kernel_size=(3, 3),dr_rate=dr_rate,
                                  strides=init_strides)(input)
        asname=name+'_'+str(ids+1)
        residual = _bn_relu_conv(filters=filters,name=asname, kernel_size=(3, 3),dr_rate=dr_rate)(conv1)
        return _shortcut(input, residual,dr_rate=False)

    return f


def bottleneck(filters, dr_rate,name,init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    Returns:
        A final conv layer of filters * 4
    """
    def f(input):
        ids=1
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            asname=name+'_'+str(ids)
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4),name=asname)(input)
            if dr_rate:
                conv_1_1 = Dropout(dr_rate)(conv_1_1, training=True)            
        else:
            asname=name+'_'+str(ids)
            conv_1_1 = _bn_relu_conv(filters=filters, name=asname,kernel_size=(1, 1),dr_rate=dr_rate,
                                     strides=init_strides)(input)
        asname=name+'_'+str(ids+1)
        conv_3_3 = _bn_relu_conv(filters=filters, name=asname,kernel_size=(3, 3),dr_rate=dr_rate)(conv_1_1)
        asname=name+'_'+str(ids+2)
        residual = _bn_relu_conv(filters=filters * 4,name=asname, kernel_size=(1, 1),dr_rate=dr_rate)(conv_3_3)
        return _shortcut(input, residual,dr_rate)

    return f


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, 
              nb_classes, 
              dropout_rate,
              model_type,
              pred_activation,
              block_fn, 
              repetitions):
        """Builds a custom ResNet like architecture.
        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved
        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[0],input_shape[1], input_shape[2])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)
        
        dr_rate=0.05
        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2),dr_rate=False,name='init_conv')(input)
        if dropout_rate:
            x = Dropout(dr_rate)(conv1, training=True)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

        block = pool1
        filters = 64
        
        for i, r in enumerate(repetitions):
            name='blk'+str(i+1)
            if dropout_rate:
                dr_rate+=0.02
            else:
                dr_rate=False
            block = _residual_block(block_fn,dropout_rate=dr_rate,filters=filters, name=name,repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)
        x = layers.GlobalAveragePooling2D()(block)
        if model_type!='class':
            x = Dense(
                nb_classes,
                activation=pred_activation,
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4),
                bias_regularizer=l2(1e-4),
            )(x)
        else:
            x = Dense(
                nb_classes,
                activation="softmax",
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4),
                bias_regularizer=l2(1e-4),
            )(x)

        model = Model(inputs=input, outputs=x)
        return model

    @staticmethod
    def build_resnet_18(input_shape, nb_classes,model_type,pred_activation,dropout_rate):
        return ResnetBuilder.build(input_shape,nb_classes,dropout_rate, model_type,pred_activation, basic_block, [2, 2, 2, 2])

    @staticmethod
    def build_resnet_34(input_shape, nb_classes,model_type,pred_activation,dropout_rate):
        return ResnetBuilder.build(input_shape, nb_classes,dropout_rate, model_type,pred_activation,basic_block, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, nb_classes,model_type,pred_activation,dropout_rate):
        return ResnetBuilder.build(input_shape, nb_classes,dropout_rate, model_type,pred_activation,bottleneck, [3, 4, 6, 3])

def resnet_like_non_temporal(
    img_row,
    img_col,
    channel,
    nb_classes=3,
    dropout_rate=True,
    which_resnet=18,
    model_type='class',
    pred_activation='tanh'
):
    function_mappings = {
    "18": ResnetBuilder.build_resnet_18,
    "34": ResnetBuilder.build_resnet_34,
    "50": ResnetBuilder.build_resnet_50
    }
    if any(which_resnet in [18,34,50] for n in [18,34,50]):
        model_name=str(which_resnet)
    else:
        raise Exception(
            "Please define number of depth (e.g. depth=18,34,50). This is required for model compilation."
        )

    return function_mappings[model_name](input_shape=(img_row,img_col,channel),
                                        nb_classes=nb_classes,
                                        dropout_rate=dropout_rate,
                                        model_type=model_type,
                                        pred_activation=pred_activation)

if __name__ == "__main__":
    resnet_like_non_temporal
    ResnetBuilder
    _handle_dim_ordering
    _bn_relu  
    _conv_bn_relu  
    _bn_relu_conv  
    _shortcut  
    _residual_block 
    basic_block 
    bottleneck 
    _get_block 
    
'''
    use case:
        model=resnet_like_non_temporal(img_row=120,
        img_col=120,
        channel=3,
        nb_classes=3,
        dropout_rate=True,
        which_resnet=18,
        model_type='class',
        pred_activation='tanh')
        #model.summary()
'''
