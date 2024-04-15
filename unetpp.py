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
from keras.layers import concatenate, Conv3D, Conv3DTranspose, Conv2D, Conv2DTranspose, Dropout, ReLU, \
    BatchNormalization, Activation
# from keras.layers.merge import add, multiply
from keras.layers import add, multiply

from keras.models import Model
from keras.optimizers import Adam
from numpy import random
from random import randint
from utils import data_augmentation, prepare_dataset
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, \
    Conv2D, Add, Activation, Lambda
from keras import backend as K
from keras.activations import sigmoid
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Add, Multiply, Concatenate, Dense, \
    MaxPooling2D


class DoubleConv(tf.keras.layers.Layer):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(out_ch, 3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(out_ch, 3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

    def call(self, inputs):
        return self.conv(inputs)


def unetpp():
    inputs = Input((None, None, 3))

    pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

    up = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')

    nb_filter = [32, 64, 128, 256, 512]
    conv0_0 = DoubleConv(in_ch=3, out_ch=nb_filter[0])
    conv1_0 = DoubleConv(in_ch=nb_filter[0], out_ch=nb_filter[1])
    conv2_0 = DoubleConv(in_ch=nb_filter[1], out_ch=nb_filter[2])
    conv3_0 = DoubleConv(in_ch=nb_filter[2], out_ch=nb_filter[3])
    conv4_0 = DoubleConv(in_ch=nb_filter[3], out_ch=nb_filter[4])

    conv0_1 = DoubleConv(in_ch=nb_filter[0] + nb_filter[1], out_ch=nb_filter[0])
    conv1_1 = DoubleConv(in_ch=nb_filter[1] + nb_filter[2], out_ch=nb_filter[1])
    conv2_1 = DoubleConv(in_ch=nb_filter[2] + nb_filter[3], out_ch=nb_filter[2])
    conv3_1 = DoubleConv(in_ch=nb_filter[3] + nb_filter[4], out_ch=nb_filter[3])

    conv0_2 = DoubleConv(in_ch=nb_filter[0] * 2 + nb_filter[1], out_ch=nb_filter[0])
    conv1_2 = DoubleConv(in_ch=nb_filter[1] * 2 + nb_filter[2], out_ch=nb_filter[1])
    conv2_2 = DoubleConv(in_ch=nb_filter[2] * 2 + nb_filter[3], out_ch=nb_filter[2])

    conv0_3 = DoubleConv(in_ch=nb_filter[0] * 3 + nb_filter[1], out_ch=nb_filter[0])
    conv1_3 = DoubleConv(in_ch=nb_filter[1] * 3 + nb_filter[2], out_ch=nb_filter[1])

    conv0_4 = DoubleConv(in_ch=nb_filter[0] * 4 + nb_filter[1], out_ch=nb_filter[0])

    final1 = tf.keras.layers.Conv2D(32, 1)

    x0_0 = conv0_0(inputs)
    x1_0 = conv1_0(pool(x0_0))
    x0_1 = conv0_1(Concatenate(axis=-1)([x0_0, up(x1_0)]))

    x2_0 = conv2_0(pool(x1_0))
    x1_1 = conv1_1(Concatenate(axis=-1)([x1_0, up(x2_0)]))
    x0_2 = conv0_2(Concatenate(axis=-1)([x0_0, x0_1, up(x1_1)]))

    x3_0 = conv3_0(pool(x2_0))
    x2_1 = conv2_1(Concatenate(axis=-1)([x2_0, up(x3_0)]))
    x1_2 = conv1_2(Concatenate(axis=-1)([x1_0, x1_1, up(x2_1)]))
    x0_3 = conv0_3(Concatenate(axis=-1)([x0_0, x0_1, x0_2, up(x1_2)]))

    x4_0 = conv4_0(pool(x3_0))
    x3_1 = conv3_1(Concatenate(axis=-1)([x3_0, up(x4_0)]))
    x2_2 = conv2_2(Concatenate(axis=-1)([x2_0, x2_1, up(x3_1)]))
    x1_3 = conv1_3(Concatenate(axis=-1)([x1_0, x1_1, x1_2, up(x2_2)]))
    x0_4 = conv0_4(Concatenate(axis=-1)([x0_0, x0_1, x0_2, x0_3, up(x1_3)]))

    output1 = final1(x0_4)
    outs = []
    out2 = Conv2D(1, (1, 1), activation='sigmoid', name='final_out')(output1)
    outs.append(out2)
    model = Model(inputs=[inputs], outputs=outs)
    loss_funcs = {}

    loss_funcs.update({'final_out': losses.binary_crossentropy})

    metrics = {
        "final_out": ['accuracy']
    }

    model.compile(optimizer=Adam(lr=1e-3), loss=loss_funcs, metrics=metrics)

    return model


def random_crop(img, mask, crop_size):
    imgheight = img.shape[0]
    imgwidth = img.shape[1]

    i = randint(0, imgheight - crop_size)
    j = randint(0, imgwidth - crop_size)

    return img[i:(i + crop_size), j:(j + crop_size), :], mask[i:(i + crop_size), j:(j + crop_size)]

#原始的
# class Generator():
#     def __init__(self, batch_size, repeat, dataset):
#         self.lock = threading.Lock()
#         self.dataset = dataset
#         with self.lock:
#             self.list_images_all = prepare_dataset.getTrainingData(0, self.dataset)
#             self.list_gt_all = prepare_dataset.getTrainingData(1, self.dataset)
#         self.n = len(self.list_images_all)
#         self.index = 0
#         self.repeat = repeat
#         self.batch_size = batch_size
#         self.step = self.batch_size // self.repeat
#
#         if self.repeat >= self.batch_size:
#             self.repeat = self.batch_size
#             self.step = 1
#
#     def gen(self, au=True, crop_size=48, iteration=None):
#
#         while True:
#             data_yield = [self.index % self.n,
#                           (self.index + self.step) % self.n if (self.index + self.step) < self.n else self.n]
#             self.index = (self.index + self.step) % self.n
#
#             list_images_base = self.list_images_all[data_yield[0]:data_yield[1]]
#             list_gt_base = self.list_gt_all[data_yield[0]:data_yield[1]]
#
#             list_images_aug = []
#             list_gt_aug = []
#             for image, gt in zip(list_images_base, list_gt_base):
#                 if au:
#                     if crop_size == prepare_dataset.DESIRED_DATA_SHAPE[0]:
#                         for _ in range(self.repeat):
#                             image, gt = data_augmentation.random_augmentation(image, gt)
#                             list_images_aug.append(image)
#                             list_gt_aug.append(gt)
#                     else:
#                         image, gt = data_augmentation.random_augmentation(image, gt)
#                         list_images_aug.append(image)
#                         list_gt_aug.append(gt)
#                 else:
#                     list_images_aug.append(image)
#                     list_gt_aug.append(gt)
#
#             list_images = []
#             list_gt = []
#
#             if crop_size == prepare_dataset.DESIRED_DATA_SHAPE[0]:
#                 list_images = list_images_aug
#                 list_gt = list_gt_aug
#             else:
#                 for image, gt in zip(list_images_aug, list_gt_aug):
#                     for _ in range(self.repeat):
#                         image_, gt_ = random_crop(image, gt, crop_size)
#
#                         list_images.append(image_)
#                         list_gt.append(gt_)
#
#             outs = {}
#             outs.update({'final_out': np.array(list_gt)})
#             yield np.array(list_images), outs

#STARE
class Generator():
    def __init__(self, batch_size, repeat, dataset,train_images,train_gt):
        self.lock = threading.Lock()
        self.dataset = dataset
        with self.lock:
            self.list_images_all = train_images
            self.list_gt_all = train_gt
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