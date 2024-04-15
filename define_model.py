import h5py
import keras.backend as K
import numpy as np
import os
import os.path
import tensorflow as tf
import threading
from PIL import Image
from keras import backend as K
from keras import losses
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Input, MaxPooling3D, MaxPooling2D
from keras.layers import concatenate, Conv3D, Conv3DTranspose, Conv2D, Conv2DTranspose, Dropout, ReLU, BatchNormalization, Activation
# from keras.layers.merge import add, multiply
from keras.layers import add, multiply
from keras.layers import DepthwiseConv2D, SeparableConv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from keras import backend as K
from keras.activations import sigmoid
from keras.models import Model
from keras.optimizers import Adam
from numpy import random
from random import randint
from utils import data_augmentation, prepare_dataset
from utils.attention import cbam_block
from utils.DropBlock2D import DropBlock2D


from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Lambda, LeakyReLU, Activation, UpSampling2D, Add, Multiply, Concatenate, Dense, MaxPooling2D
from keras.backend import mean, max, sigmoid
import tensorflow as tf
def spatial_attention(input_feature):
    kernel_size = 7

    avg_pool = Lambda(lambda x: mean(x, axis=3, keepdims=True))(input_feature)
    max_pool = Lambda(lambda x: max(x, axis=3, keepdims=True))(input_feature)
    concat = Concatenate(axis=3)([avg_pool, max_pool])

    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          use_bias=False)(concat)

    return Multiply()([input_feature, cbam_feature])
#


#
class conv_block:
    def __init__(self, ch_in, ch_out, block_size=7, keep_prob=0.9, kernel_size=3, stride=1, padding='same', use_bias=True):
        self.conv = Conv2D(ch_out, kernel_size=1, strides=1, padding='same')

        self.conv1_1 = Conv2D(ch_out, kernel_size, strides=stride, padding=padding, use_bias=use_bias)
        self.bn1 = BatchNormalization()
        self.act1 = Activation('relu')


        self.conv2_1 = Conv2D(ch_out, kernel_size, strides=stride, padding=padding, use_bias=use_bias)
        self.bn2 = BatchNormalization()
        self.act2 = Activation('relu')

    def __call__(self, x):
        x = self.conv(x)
        x_ = x
        x = self.conv1_1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2_1(x)
        x = self.bn2(x)
        x = x_ + x
        x = self.act2(x)

        return x


# #
# class conv_block:
#     def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding='same', use_bias=True):
#         self.conv = Sequential([
#             Conv2D(ch_out, kernel_size, strides=stride, padding=padding, use_bias=use_bias, input_shape=(None, None, ch_in)),
#             BatchNormalization(),
#             Activation('relu'),
#             Conv2D(ch_out, kernel_size, strides=stride, padding=padding, use_bias=use_bias),
#             BatchNormalization(),
#             Activation('relu')
#         ])
#
#     def __call__(self, x):
#         x = self.conv(x)
#         return x
class up_conv:
    def __init__(self, ch_in, ch_out, kernel_size=3, scale_factor=2, padding='same', use_bias=True):
        self.up = Sequential([
            UpSampling2D(size=(scale_factor, scale_factor)),
            Conv2D(ch_out, kernel_size, strides=1, padding=padding, use_bias=use_bias),
            BatchNormalization(),
            Activation('relu')
        ])

    def __call__(self, x):
        return self.up(x)
#
class Attention_block:
    def __init__(self, F_g, F_l, F_int):
        self.W_g = Sequential([
            Conv2D(F_int, kernel_size=1, strides=1, padding='same', use_bias=True),
            BatchNormalization()
        ])

        self.W_x = Sequential([
            Conv2D(F_int, kernel_size=1, strides=1, padding='same', use_bias=True),
            BatchNormalization()
        ])

        self.psi = Sequential([
            Conv2D(1, kernel_size=1, strides=1, padding='same', use_bias=True),
            BatchNormalization(),
            Activation('sigmoid')
        ])


        # 使用Dense层定义可学习参数 weight_g
        self.weight_g = Dense(F_g, use_bias=True, activation='linear', input_shape=(1, 1, 1, F_g * 2))
        #
        self.conv_ = Conv2D(F_g, kernel_size=1, strides=1, padding='same')
        self.relu = Activation('relu')

    def __call__(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(Add()([g1, x1]))

        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        weight_g = Multiply()([g, self.weight_g(g)])
        x_p = Multiply()([x, psi])
        a = Concatenate(axis=-1)([weight_g, x_p])
        a = self.conv_(a)
        return a
#

# class Attention_block:
#     def __init__(self, F_g, F_l, F_int):
#         self.W_g = Sequential([
#             Conv2D(F_int, kernel_size=1, strides=1, padding='same', use_bias=True),
#             BatchNormalization()
#         ])
#
#         self.W_x = Sequential([
#             Conv2D(F_int, kernel_size=1, strides=1, padding='same', use_bias=True),
#             BatchNormalization()
#         ])
#
#         self.psi = Sequential([
#             Conv2D(1, kernel_size=1, strides=1, padding='same', use_bias=True),
#             BatchNormalization(),
#             Activation('sigmoid')
#         ])
#
#
#         # 使用Dense层定义可学习参数 weight_g
#         self.weight_g = Dense(F_g, use_bias=True, activation='linear', input_shape=(1.5, 1.5, 1.5, F_g * 2))
#         #
#         self.conv_ = Conv2D(F_g, kernel_size=1, strides=1, padding='same')
#         self.relu = Activation('relu')
#
#     def __call__(self, g, x):
#         # 下采样的gating signal 卷积
#         g1 = self.W_g(g)
#         # 上采样的 l 卷积
#         x1 = self.W_x(x)
#         # concat + relu
#         psi = self.relu(Add()([g1, x1]))
#
#         # channel 减为1，并Sigmoid,得到权重矩阵
#         psi = self.psi(psi)
#         # 返回加权的 x
#         # weight_g = Multiply()([g, self.weight_g(g)])
#         x_p = Multiply()([x, psi])
#         a = Concatenate(axis=-1)([1.5*g, x_p])
#         a = self.conv_(a)
#         return a



# 原始的函数
def get_Attention():
    inputs = Input((None, None, 3))
    ##############原始的#############
    conv1 = conv_block(ch_in=3, ch_out=64)
    conv2 = conv_block(ch_in=64, ch_out=128)
    conv3 = conv_block(ch_in=128, ch_out=256)
    conv4 = conv_block(ch_in=256, ch_out=512)
    conv5 = conv_block(ch_in=512, ch_out=1024)
    ##############原始的#############


    Up5 = up_conv(ch_in=1024, ch_out=512)
    Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
    Up_conv5 = conv_block(ch_in=1024, ch_out=512)

    Up4 = up_conv(ch_in=512, ch_out=256)
    Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
    Up_conv4 = conv_block(ch_in=512, ch_out=256)

    Up3 = up_conv(ch_in=256, ch_out=128)
    Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
    Up_conv3 = conv_block(ch_in=256, ch_out=128)

    Up2 = up_conv(ch_in=128, ch_out=64)
    Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
    Up_conv2 = conv_block(ch_in=128, ch_out=64)

    x1 = conv1(inputs)
    x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x1)
    x2 = conv2(x2)

    x3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x2)
    x3 = conv3(x3)

    x4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x3)
    x4 = conv4(x4)

    x5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x4)
    x5 = conv5(x5)
    # x5 = spatial_attention(x5)
    d5 = Up5(x5)
    x4 = Att5(g=d5, x=x4)
    d5 = Concatenate(axis=-1)([x4, d5])
    d5 = Up_conv5(d5)

    d4 = Up4(d5)
    x3 = Att4(g=d4, x=x3)
    d4 = Concatenate(axis=-1)([x3, d4])
    d4 = Up_conv4(d4)

    d3 = Up3(d4)
    x2 = Att3(g=d3, x=x2)
    d3 = Concatenate(axis=-1)([x2, d3])
    d3 = Up_conv3(d3)

    d2 = Up2(d3)
    x1 = Att2(g=d2, x=x1)
    d2 = Concatenate(axis=-1)([x1, d2])
    d2 = Up_conv2(d2)

    outs = []

    out2 = Conv2D(1, (1, 1), activation='sigmoid', name='final_out')(d2)
    outs.append(out2)
    model = Model(inputs=[inputs], outputs=outs)
    loss_funcs = {}

    loss_funcs.update({'final_out': losses.binary_crossentropy})


    metrics = {
        "final_out": ['accuracy']
    }

    model.compile(optimizer=Adam(lr=1e-3), loss=loss_funcs, metrics=metrics)

    return model


# 函数的主要目的是从给定的图像和掩码中随机裁剪出一个指定大小的区域
def random_crop(img, mask, crop_size):
    imgheight = img.shape[0]
    imgwidth = img.shape[1]

    i = randint(0, imgheight - crop_size)
    j = randint(0, imgwidth - crop_size)

    return img[i:(i + crop_size), j:(j + crop_size), :], mask[i:(i + crop_size), j:(j + crop_size)]

# 原始的
class Generator():
    def __init__(self, batch_size, repeat, dataset):
        self.lock = threading.Lock()
        self.dataset = dataset
        with self.lock:
            self.list_images_all = prepare_dataset.getTrainingData(0, self.dataset)
            self.list_gt_all = prepare_dataset.getTrainingData(1, self.dataset)
        self.n = len(self.list_images_all)
        self.index = 0
        self.repeat = repeat
        self.batch_size = batch_size
        self.step = self.batch_size // self.repeat

        if self.repeat >= self.batch_size:
            self.repeat = self.batch_size
            self.step = 1

    def gen(self, au=True, crop_size=48, iteration=None):

        while True:
            data_yield = [self.index % self.n,
                          (self.index + self.step) % self.n if (self.index + self.step) < self.n else self.n]
            self.index = (self.index + self.step) % self.n

            list_images_base = self.list_images_all[data_yield[0]:data_yield[1]]
            list_gt_base = self.list_gt_all[data_yield[0]:data_yield[1]]

            list_images_aug = []
            list_gt_aug = []
            # 随机裁剪和数据增强只能有一个重复 repeat 次
            for image, gt in zip(list_images_base, list_gt_base):
                if au:
                    if crop_size == prepare_dataset.DESIRED_DATA_SHAPE[0]:
                        for _ in range(self.repeat):
                            image, gt = data_augmentation.random_augmentation(image, gt)
                            list_images_aug.append(image)
                            list_gt_aug.append(gt)
                    else:
                        image, gt = data_augmentation.random_augmentation(image, gt)
                        list_images_aug.append(image)
                        list_gt_aug.append(gt)
                else:
                    list_images_aug.append(image)
                    list_gt_aug.append(gt)

            list_images = []
            list_gt = []

            if crop_size == prepare_dataset.DESIRED_DATA_SHAPE[0]:
                list_images = list_images_aug
                list_gt = list_gt_aug
            else:
                for image, gt in zip(list_images_aug, list_gt_aug):
                    for _ in range(self.repeat):
                        image_, gt_ = random_crop(image, gt, crop_size)

                        list_images.append(image_)
                        list_gt.append(gt_)

            outs = {}
            # for iteration_id in range(iteration):
            #     outs.update({f'out1{iteration_id + 1}': np.array(list_gt)})
            outs.update({'final_out': np.array(list_gt)})
            yield np.array(list_images), outs

