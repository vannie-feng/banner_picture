#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from sys import exit
import os
import numpy as np
from collections import OrderedDict
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.applications import resnet50, vgg16, vgg19, inception_v3, xception
from keras.applications import inception_resnet_v2, mobilenet, densenet, mobilenet_v2
resnet50.ResNet50.__parent__ = resnet50
vgg16.VGG16.__parent__ = vgg16
vgg19.VGG19.__parent__ = vgg19
inception_v3.InceptionV3.__parent__ = inception_v3
xception.Xception.__parent__ = xception
inception_resnet_v2.InceptionResNetV2.__parent__ = inception_resnet_v2
mobilenet.MobileNet.__parent__ = mobilenet
densenet.DenseNet121.__parent__ = densenet
densenet.DenseNet169.__parent__ = densenet
densenet.DenseNet201.__parent__ = densenet
mobilenet_v2.MobileNetV2.__parent__ = mobilenet_v2
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l1, l2, l1_l2
from keras.utils import to_categorical
from keras_metrics import f1_score

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
TRAIN = 'train'
VALIDATION = 'validation'
TEST = 'test'
FEATURE_PREF = 'bottleneck_features'
CLASSES = ['0', '1']
top_model_weights_path = 'bottleneck_fc_model.h5'
model_weights_path = 'finetune_model.h5'

img_height, img_width= 224, 224
epochs = 40
finetune_epochs = 30
batch_size = 16
# freeze_count = 0
# freeze_count = 7
# freeze_count = 39
# freeze_count = 81
# freeze_count = 143
freeze_count = 175
is_final = False
saved = False

BASE_MODEL = resnet50.ResNet50
# BASE_MODEL = vgg16.VGG16
# BASE_MODEL = vgg19.VGG19
# BASE_MODEL = inception_v3.InceptionV3
# BASE_MODEL = xception.Xception
# BASE_MODEL = inception_resnet_v2.InceptionResNetV2
# BASE_MODEL = mobilenet.MobileNet
# BASE_MODEL = densenet.DenseNet121
# BASE_MODEL = densenet.DenseNet169
# BASE_MODEL = densenet.DenseNet201
# BASE_MODEL = mobilenet_v2.MobileNetV2


def gen_top_model(input_shape, direct=True):

    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=input_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid',
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01)))
    if direct:
        model.compile(optimizer=SGD(momentum=0.5),
                loss='binary_crossentropy',
                metrics=['accuracy', f1_score(label=1)])

    return model

class DataItem():

    pref = ''

    def __init__(self, data_dir):

        self.data_dir = data_dir

    def gen_feat_file(self, key):

        return '%s_%s.npy' % (self.__class__.pref, key)

DataItem.pref = FEATURE_PREF

def base_model_conf():

    return (BASE_MODEL.__parent__.preprocess_input, BASE_MODEL)

def trans_bottleneck_features(data_map):

    if saved:
        return
    preprocess_input, base_model_type = base_model_conf()
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    model = base_model_type(include_top=False, weights='imagenet')
    print('Model loaded.')

    def trans_feature(data_item, key):
    
        batch_size = 1
        generator = datagen.flow_from_directory(
            data_item.data_dir,
            classes=CLASSES,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False,
            follow_links=True)

        bottleneck_features = model.predict_generator(
            generator, data_item.sample_count // batch_size, workers=16)
        np.save(open(data_item.gen_feat_file(key), 'w'), bottleneck_features)

    for key, data_item in data_map.iteritems():

        trans_feature(data_item, key)

def parse_data_item(data_map, key):

    data_item = data_map[key]
    ffeat_name = data_item.gen_feat_file(key)
    return np.load(open(ffeat_name)), data_item.labels

def train_top_model(data_map):

    train_data, train_labels = parse_data_item(data_map, TRAIN)
    input_shape = train_data.shape[1:]
    print input_shape
    model = gen_top_model(input_shape)
    validation_data = parse_data_item(data_map, VALIDATION)
    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=validation_data)
    model.save_weights(top_model_weights_path)

def eval_top_model(data_map):

    test_data, test_labels = parse_data_item(data_map, TEST)
    input_shape = test_data.shape[1:]
    print input_shape
    model = gen_top_model(input_shape)
    model.load_weights(top_model_weights_path)
    res = model.evaluate(test_data,
            test_labels,
            batch_size=batch_size)
    print zip(model.metrics_names, res)

def trial_top_model(data_map):

    trans_bottleneck_features(data_map)
    train_top_model(data_map)
    eval_top_model(data_map)

def check_data_map(data_map):

    for key, data_item in data_map.iteritems():

        data_dir = data_item.data_dir
        data_dir += '' if data_dir.endswith('/') else '/'
        n = len(data_dir)
        assert os.path.exists(data_dir)
        cls_counts = [0] * len(CLASSES)
        for root, dirs, files in os.walk(data_dir, followlinks=True):

            if root == data_dir:
                continue
            cls = root[n:].split(os.sep)[0]
            idx = CLASSES.index(cls)
            for name in files:

                cls_counts[idx] += 1

        data_item.sample_count = sum(cls_counts)
        labels = []
        for i, x in enumerate(cls_counts):

            labels += [i] * x

        data_item.labels = np.array(labels, dtype='int32')

def gen_model():

    preprocess_input, base_model_type = base_model_conf()
    base_model = base_model_type(weights='imagenet', include_top=False,
            input_shape=(img_height, img_width, 3), freeze_count=freeze_count)
    print freeze_count
    for layer in base_model.layers[:freeze_count]:

        layer.trainable = False

    for i, layer in enumerate(base_model.layers):

        print i, layer.name, layer.trainable

    print('Model loaded.')
    input_shape = base_model.output_shape[1:]
    print input_shape
    top_model = gen_top_model(input_shape, False)
    top_model.load_weights(top_model_weights_path)
    model = Model(inputs=base_model.inputs, outputs=top_model(base_model.outputs))
    # model.load_weights(model_weights_path) # for temp
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
            loss='binary_crossentropy',
            metrics=['accuracy', f1_score(label=1)])

    return preprocess_input, model

def finetune(data_map):
    
    key = VALIDATION
    # key = TEST
    train_steps = data_map[TRAIN].sample_count // batch_size
    preprocess_input, model = gen_model()
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=(0.3, 1.0),
        shear_range=0.2,
        zoom_range=0.2,
        channel_shift_range=10.0,
        horizontal_flip=True,
        vertical_flip=True)

    def trans_feature(datagen, data_item):
    
        return datagen.flow_from_directory(
            data_item.data_dir,
            classes=CLASSES,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='binary',
            follow_links=True)

    train_generator = trans_feature(train_datagen, data_map[TRAIN])
    validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    validation_generator = trans_feature(validation_datagen, data_map[key])
    validation_data = validation_generator
    validation_steps = data_map[key].sample_count // batch_size
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=finetune_epochs,
        validation_data=validation_data,
        validation_steps=validation_steps,
        workers=16)
    model.save_weights(model_weights_path)

def eval_model(data_map):

    test_steps = data_map[TEST].sample_count // batch_size
    preprocess_input, model = gen_model()
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    def trans_feature(datagen, data_item):
    
        return datagen.flow_from_directory(
            data_item.data_dir,
            classes=CLASSES,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='binary',
            follow_links=True)

    test_generator = trans_feature(test_datagen, data_map[TEST])
    model.load_weights(model_weights_path)
    res = model.evaluate_generator(test_generator, test_steps, workers=16)
    print zip(model.metrics_names, res)

def trial_finetune(data_map):

    finetune(data_map)
    eval_model(data_map)


if __name__ == '__main__':

    if is_final:
        data_map = OrderedDict([(TRAIN, DataItem('data/train.final')),
            (VALIDATION, DataItem('data/test')),
            (TEST, DataItem('data/test'))])
    else:
        data_map = OrderedDict([(TRAIN, DataItem('data/train')),
            (VALIDATION, DataItem('data/validate')),
            (TEST, DataItem('data/test'))])
    check_data_map(data_map)
    trial_top_model(data_map)
    # trial_finetune(data_map)

    exit(0)
