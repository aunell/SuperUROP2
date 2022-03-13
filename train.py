#Baseline
# from __future__ import print_function

import argparse

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, GaussianNoise
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray, gray2rgb
from tensorflow.keras import datasets, layers, models, backend, Model, callbacks

import wandb
import tensorflow as tf


wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 128
}
run = wandb.init(project='devNoise', config=wandb.config, entity="aunell")
config = wandb.config

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--experiment_id", type=int, default=1)
#index 0= {0=baseline, 1=grayblurcolor, 2=gbnoise, 3=bio, 4=antibio, 5=noise, 6=gb, 7=all}
#index1= {0=alex, 1=resnet}
args = parser.parse_args() #args should be id

args=str(args.experiment_id)

# Training parameters
batch_size = 32  # orig paper trained all networks with batch_size=128
epochs = 200  # 200 gives optimal acc
data_augmentation = False
num_classes = 10

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
n = 6

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 2

# Computed depth from supplied model parameter n
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2


def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r


# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)


def testData(data, imNoise=None, resnet=True, cifarIndex=0):
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_imagesClear = np.copy(train_images)
    for i in range(len(train_images)):
        image = train_imagesClear[i]
        image = cv2.GaussianBlur(image, (3, 3), 0)
    train_imagesClear = train_imagesClear / 255.0
    for i in range(len(train_imagesClear)):
        image = rgb2gray(train_imagesClear[i])
        image = gray2rgb(image)
        train_imagesClear[i] = image
    train_images = train_images / 255.0
    # train_imagesClear=train_imagesClear/255.0
    print('train shape', train_images.shape)
    if data == 'cifar10':
        test_images = test_images / 1.0
        if imNoise != None:
            testNoise = imNoise / 100 * 255
            test_images = np.copy(test_images)
            for i in range(len(test_images)):
                gauss = np.random.normal(0, testNoise, (32, 32, 3))
                gauss = gauss.reshape(32, 32, 3)
                image = (test_images[i] + gauss)
                image = np.clip(image, 0, 255)
                test_images[i] = image
    else:
        dirs = list_files('/Users/alyssaunell/Desktop/Desktop/SuperUROP/CIFAR-10-C') #need to get Cifar10c on supercomp
        test_labels = np.load('/Users/alyssaunell/Desktop/Desktop/SuperUROP/CIFAR-10-C/labels.npy')
        test_labels = np.array([test_labels])
        test_labels = test_labels.transpose()
        test_labels = test_labels[-10000:]
        test_images = np.load(dirs[cifarIndex])
        test_images = test_images[-10000:]
        input_shape = test_images.shape[1:]
    # Normalize pixel values to be between 0 and 1
    test_images = test_images / 255.0
    if resnet:
        train_labels = keras.utils.to_categorical(train_labels, num_classes)
        test_labels = keras.utils.to_categorical(test_labels, num_classes)
        if subtract_pixel_mean:
            x_test_mean = np.mean(test_images, axis=0)
            test_images -= x_test_mean

            train_imagesMean = np.mean(train_images, axis=0)
            train_images -= train_imagesMean

            train_imagesClearMean = np.mean(train_imagesClear, axis=0)
            train_imagesClear -= train_imagesClearMean
    # for i in range(3):
    #     plt.subplot(5, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_images[i + 5], cmap='gray')
    #     # The CIFAR labels happen to be arrays,
    #     # which is why you need the extra index
    #     plt.xlabel('reg')
    # plt.show()
    return [[train_images, train_labels], [train_imagesClear, train_labels], [test_images, test_labels]]


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v2(input_shape, depth, num_classes=10, noise=False):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2  # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    if noise:
        x = GaussianNoise(.3)(x)
        # loc 1
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    #     if noise:
    #         x = GaussianNoise(.1)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


# Input image dimensions.
# input_shape = testData('cifar10', imNoise=0, resnet=True, cifarIndex=0)[0]
# print(len(input_shape))
# input_shape=np.array(input_shape)
(train_images, train_labels2), (test_images2, test_labels2) = cifar10.load_data()
input_shape = train_images.shape[1:]

model = resnet_v2(input_shape=input_shape, depth=depth, noise=True)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()
print(model_type)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [lr_reducer, lr_scheduler]

##RUN TRAINING
baseline = False
grayBlurColor = False
grayBlurNoise = False
bioMimetic = False
antiBio = False
noise = False
grayBlur = False
trainRes=False
trainAlex=False
import tensorflow as tf
if args[0]=="7":
    baseline = True
    grayBlurColor = True
    grayBlurNoise = True
    bioMimetic = True
    antiBio = True
    noise = True
    grayBlur = True
if args[0]=="0":
    baseline = True
if args[0]=="1":
    grayBlurColor = True
if args[0]=="2":
    grayBlurNoise = True
if args[0]=="3":
    bioMimetic = True
if args[0]=="4":
    antiBio = True
if args[0]=="5":
    noise = True
if args[0]=="6":
    grayBlur = True
if args[1]=="0":
    trainAlex = True
if args[1]=="1":
    trainRes = True

data = testData('cifar10', imNoise=None, resnet=trainRes, cifarIndex=0)

validate_images = data[2][0][0:2000]
validate_labels = data[2][1][0:2000]

train_images = data[0][0]
train_labels = data[0][1]

train_imagesClear = data[1][0]
train_labelsClear = data[1][1]

if trainRes:
    if baseline:
        # baseline
        model = resnet_v2(input_shape=input_shape, depth=depth, noise=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=lr_schedule(0)),
                      metrics=['accuracy'])
        history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs,
                            validation_data=(validate_images, validate_labels), callbacks=callbacks)
        name = 'baseline'
        path = '/Users/alyssaunell/Desktop/Desktop/SuperUROP/res/models/' + name
        model.save(path)
    if grayBlur or bioMimetic or grayBlurColor:
        modelGray = resnet_v2(input_shape=input_shape, depth=depth, noise=False)
        modelGray.compile(loss='categorical_crossentropy',
                          optimizer=Adam(lr=lr_schedule(0)),
                          metrics=['accuracy'])
        history = modelGray.fit(train_imagesClear, train_labels, batch_size=batch_size, epochs=epochs,
                                validation_data=(validate_images, validate_labels), callbacks=callbacks)
        name = 'grayBlurloc1'
        path = path = '/Users/alyssaunell/Desktop/Desktop/SuperUROP/res/models/' + name
        if grayBlur:
            modelGray.save(path)
    if bioMimetic:
        weights = modelGray.get_weights()
        model = None
        print('starting biomodel')
        model = resnet_v2(input_shape=input_shape, depth=depth, noise=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=lr_schedule(0)),
                      metrics=['accuracy'])
        model.set_weights(weights)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=50, mode='max')
        callbacks.append(callback)
        history = model.fit(train_images, train_labels, batch_size=batch_size,
                            validation_data=(validate_images, validate_labels), epochs=100,
                            callbacks=callbacks)

        print('ending biomodel')
        name = 'bioMimetic'
        path = path = '/Users/alyssaunell/Desktop/Desktop/SuperUROP/res/models/' + name
        model.save(path)
        callbacks.pop()

    if grayBlurColor:
        # model = tf.keras.models.load_model('/om/user/aunell/data/grayBlurloc1')
        weights = modelGray.get_weights()
        model = None
        print('starting biomodel')
        model = resnet_v2(input_shape=input_shape, depth=depth, noise=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=lr_schedule(0)),
                      metrics=['accuracy'])
        model.set_weights(weights)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=50, mode='max')
        callbacks.append(callback)
        history = model.fit(train_images, train_labels, batch_size=batch_size,
                            validation_data=(validate_images, validate_labels), epochs=100,
                            callbacks=callbacks)

        print('ending biomodel')
        name = 'grayBlurColor'
        path = '/Users/alyssaunell/Desktop/Desktop/SuperUROP/res/models/' + name
        model.save(path)
        callbacks.pop()
    if grayBlurNoise:
        model = resnet_v2(input_shape=input_shape, depth=depth, noise=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=lr_schedule(0)),
                      metrics=['accuracy'])
        history = model.fit(train_imagesClear, train_labels, batch_size=batch_size, epochs=epochs,
                            validation_data=(validate_images, validate_labels), callbacks=callbacks)
        name = 'grayBlurNoise'
        path = '/om/user/aunell/restruc/res/' + name
        model.save(path)
    if antiBio or noise:
        model = resnet_v2(input_shape=input_shape, depth=depth, noise=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=lr_schedule(0)),
                      metrics=['accuracy'])
        history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs,
                            validation_data=(validate_images, validate_labels), callbacks=callbacks)
        name = 'noise'
        path = '/Users/alyssaunell/Desktop/Desktop/SuperUROP/res/models/' + name
        if noise:
            model.save(path)
        if antiBio:
            weights = model.get_weights()
            model = None
            model = resnet_v2(input_shape=input_shape, depth=depth, noise=False)
            model.compile(loss='categorical_crossentropy',
                          optimizer=Adam(lr=lr_schedule(0)),
                          metrics=['accuracy'])
            model.set_weights(weights)
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=50, mode='max')
            callbacks.append(callback)
            history = model.fit(train_imagesClear, train_labels, batch_size=batch_size,
                                validation_data=(validate_images, validate_labels), epochs=100,
                                callbacks=callbacks)
            name = 'antiBio'
            path = '/Users/alyssaunell/Desktop/Desktop/SuperUROP/res/models/' + name
            model.save(path)
    wandb.tensorflow.log(tf.summary.merge_all())
if trainAlex:
    trainNoise = .1
    if baseline:
        for i in range(1, 4):
            noise_dict = {1: 0, 2: 0, 3: 0}
            if i != 1:
                weights0 = model.get_weights()
                del (model)
                tf.compat.v1.reset_default_graph()
            print('starting model')
            model = models.Sequential()
            model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
            model.add(layers.Activation('relu'))
            model.add(layers.GaussianNoise(noise_dict[1]))
            model.add(layers.MaxPooling2D((2, 2)))

            model.add(layers.Conv2D(64, (3, 3)))
            model.add(layers.Activation('relu'))
            model.add(layers.GaussianNoise(noise_dict[2]))
            model.add(layers.MaxPooling2D((2, 2)))

            model.add(layers.Conv2D(64, (3, 3)))
            model.add(layers.Activation('relu'))
            model.add(layers.GaussianNoise(noise_dict[3]))
            model.add(layers.Flatten())
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(10))
            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
            if i != 1:
                model.set_weights(weights0)
            history = model.fit(train_images, train_labels, epochs=3,
                                validation_data=(validate_images, validate_labels))
        name = 'baseline'
        path = '/Users/alyssaunell/Desktop/Desktop/SuperUROP/alex/models/' + name
        model.save(path)
    if grayBlur or bioMimetic or grayBlurColor:
        for i in range(1, 4):
            noise_dict = {1: 0, 2: 0, 3: 0}
            if i != 1:
                weights0 = model.get_weights()
                del (model)
                tf.compat.v1.reset_default_graph()
            print('starting model')
            model = models.Sequential()
            model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
            model.add(layers.Activation('relu'))
            model.add(layers.GaussianNoise(noise_dict[1]))
            model.add(layers.MaxPooling2D((2, 2)))

            model.add(layers.Conv2D(64, (3, 3)))
            model.add(layers.Activation('relu'))
            model.add(layers.GaussianNoise(noise_dict[2]))
            model.add(layers.MaxPooling2D((2, 2)))

            model.add(layers.Conv2D(64, (3, 3)))
            model.add(layers.Activation('relu'))
            model.add(layers.GaussianNoise(noise_dict[3]))
            model.add(layers.Flatten())
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(10))
            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
            if i != 1:
                model.set_weights(weights0)
            history = model.fit(train_imagesClear, train_labels, epochs=3,
                                validation_data=(validate_images, validate_labels))
        if grayBlur:
            name = 'grayBlur'
            path = '/Users/alyssaunell/Desktop/Desktop/SuperUROP/alex/models/' + name
            model.save(path)
        modelGray = model
    if bioMimetic:
        weights0 = modelGray.get_weights()
        model = None
        noise_dict = {1: 0, 2: trainNoise, 3: 0}
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
        model.add(layers.Activation('relu'))
        model.add(layers.GaussianNoise(noise_dict[1]))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3)))
        model.add(layers.Activation('relu'))
        model.add(layers.GaussianNoise(noise_dict[2]))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3)))
        model.add(layers.Activation('relu'))
        model.add(layers.GaussianNoise(noise_dict[3]))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10))
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        model.set_weights(weights0)

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, mode='min')
        history = model.fit(train_images, train_labels,
                            validation_data=(validate_images, validate_labels), epochs=50,
                            callbacks=[callback])
        name = 'bioMimetic'
        path = '/Users/alyssaunell/Desktop/Desktop/SuperUROP/alex/models/' + name
        model.save(path)
    if grayBlurColor:
        weights0 = modelGray.get_weights()
        model = None
        noise_dict = {1: 0, 2: 0, 3: 0}
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
        model.add(layers.Activation('relu'))
        model.add(layers.GaussianNoise(noise_dict[1]))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3)))
        model.add(layers.Activation('relu'))
        model.add(layers.GaussianNoise(noise_dict[2]))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3)))
        model.add(layers.Activation('relu'))
        model.add(layers.GaussianNoise(noise_dict[3]))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10))
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        model.set_weights(weights0)

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, mode='min')
        history = model.fit(train_images, train_labels,
                            validation_data=(validate_images, validate_labels), epochs=50,
                            callbacks=[callback])
        name = 'grayBlurColor'
        path = '/Users/alyssaunell/Desktop/Desktop/SuperUROP/alex/models/' + name
        model.save(path)
    if grayBlurNoise:
        for i in range(1, 4):
            noise_dict = {1: 0, 2: trainNoise, 3: 0}
            if i != 1:
                weights0 = model.get_weights()
                del (model)
                tf.compat.v1.reset_default_graph()
            print('starting model')
            model = models.Sequential()
            model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
            model.add(layers.Activation('relu'))
            model.add(layers.GaussianNoise(noise_dict[1]))
            model.add(layers.MaxPooling2D((2, 2)))

            model.add(layers.Conv2D(64, (3, 3)))
            model.add(layers.Activation('relu'))
            model.add(layers.GaussianNoise(noise_dict[2]))
            model.add(layers.MaxPooling2D((2, 2)))

            model.add(layers.Conv2D(64, (3, 3)))
            model.add(layers.Activation('relu'))
            model.add(layers.GaussianNoise(noise_dict[3]))
            model.add(layers.Flatten())
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(10))
            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
            if i != 1:
                model.set_weights(weights0)
            history = model.fit(train_imagesClear, train_labels, epochs=3,
                                validation_data=(validate_images, validate_labels))
        name = 'grayBlurNoise'
        path = '/Users/alyssaunell/Desktop/Desktop/SuperUROP/alex/models/' + name
        model.save(path)
    if antiBio or noise:
        for i in range(1, 4):
            noise_dict = {1: 0, 2: trainNoise, 3: 0}
            if i != 1:
                weights0 = model.get_weights()
                del (model)
                tf.compat.v1.reset_default_graph()
            print('starting model')
            model = models.Sequential()
            model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
            model.add(layers.Activation('relu'))
            model.add(layers.GaussianNoise(noise_dict[1]))
            model.add(layers.MaxPooling2D((2, 2)))

            model.add(layers.Conv2D(64, (3, 3)))
            model.add(layers.Activation('relu'))
            model.add(layers.GaussianNoise(noise_dict[2]))
            model.add(layers.MaxPooling2D((2, 2)))

            model.add(layers.Conv2D(64, (3, 3)))
            model.add(layers.Activation('relu'))
            model.add(layers.GaussianNoise(noise_dict[3]))
            model.add(layers.Flatten())
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(10))
            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
            if i != 1:
                model.set_weights(weights0)
            history = model.fit(train_images, train_labels, epochs=3,
                                validation_data=(validate_images, validate_labels))
        if noise:
            name = 'noise'
            path = '/Users/alyssaunell/Desktop/Desktop/SuperUROP/alex/models/' + name
            model.save(path)
        if antiBio:
            weights0 = model.get_weights()
            model = None
            noise_dict = {1: 0, 2: 0, 3: 0}
            model = models.Sequential()
            model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
            model.add(layers.Activation('relu'))
            model.add(layers.GaussianNoise(noise_dict[1]))
            model.add(layers.MaxPooling2D((2, 2)))

            model.add(layers.Conv2D(64, (3, 3)))
            model.add(layers.Activation('relu'))
            model.add(layers.GaussianNoise(noise_dict[2]))
            model.add(layers.MaxPooling2D((2, 2)))

            model.add(layers.Conv2D(64, (3, 3)))
            model.add(layers.Activation('relu'))
            model.add(layers.GaussianNoise(noise_dict[3]))
            model.add(layers.Flatten())
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(10))
            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
            model.set_weights(weights0)

            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, mode='min')
            history = model.fit(train_imagesClear, train_labels,
                                validation_data=(validate_images, validate_labels), epochs=50,
                                callbacks=[callback])
            name = 'antiBio'
            path = '/Users/alyssaunell/Desktop/Desktop/SuperUROP/alex/models/' + name
            model.save(path)
    wandb.tensorflow.log(tf.summary.merge_all())
run.finish()