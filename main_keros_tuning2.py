import numpy as np
import tensorflow as tf
import tensorflow as tensorflow
from tensorflow import keras
import os
import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, Input, Dropout, BatchNormalization, Conv2D, MaxPooling2D 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from kerastuner.tuners import RandomSearch
#image_dataset_from_directory = tensorflow.keras.preprocessing.image_dataset_from_directory


# Каталог с данными для обучения
train_dir = 'C:\\work1\\ATM_foto\\train'
# Каталог с данными для проверки
val_dir = 'C:\\work1\\ATM_foto\\val'
# Каталог с данными для тестирования
test_dir = 'C:\\work1\\ATM_foto\\test'
# Размеры изображения
img_width, img_height = 110, 110
# Размерность тензора на основе изображения для входных данных в нейронную сеть
# backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)
# Количество эпох
epochs = 1
# Размер мини-выборки
batch_size = 32#16

# Количество изображений для обучения
nb_train_samples = len(os.listdir(train_dir+'\\atm'))
# Количество изображений для проверки
nb_validation_samples = len(os.listdir(val_dir+'\\atm'))
# Количество изображений для тестирования
nb_test_samples = len(os.listdir(test_dir+'\\atm'))

# train_dataset = image_dataset_from_directory(train_dir, batch_size=batch_size, image_size=image_size)
# validation_dataset = image_dataset_from_directory(val_dir, batch_size=batch_size, image_size=image_size)
# test_dataset = image_dataset_from_directory(test_dir, batch_size=batch_size, image_size=image_size)
train_datagen = ImageDataGenerator(rescale=1. / 255) 
test_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

print('version tensorflow-',tf.__version__)
print ("Генератор данных для обучения на основе изображений из каталога")
train_dataset = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

print ("Генератор данных для проверки на основе изображений из каталога")
validation_dataset = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

print ("Генератор данных для тестирования на основе изображений из каталога")
test_dataset = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


# AUTOTUNE = tensorflow.data.experimental.AUTOTUNE
# train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
# validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
# test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


def create_model(hp):
#    global input_shape
    model = Sequential()
    model.add(Conv2D(hp.Int(f'count_conv2d_neur_layer-input', min_value=32, max_value=96, step=32), (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
#    count_conv2d_size = hp.Int('count_conv2d_size_layer', min_value=2, max_value=5, step=1) 

    for i in range(1, hp.Int('layers_conv', 1, 5)):
        model.add(Conv2D(hp.Int(f'count_conv2d_neur_layer-{i}', min_value=32, max_value=64, step=32), (3,3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(hp.Float(f'dropout_Dense_layer-{i}', min_value=0.125, max_value=0.5, step=0.125)))

    model.add(Flatten())
    model.add(Dense(hp.Int(f'count_neur_layer-Dense-1', min_value=64, max_value=256, step=64)))
    model.add(Activation('relu'))
    model.add(Dropout(hp.Float(f'dropout_1', min_value=0.125, max_value=0.5, step=0.125)))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam','adamax']),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model   

tuner = RandomSearch(
    create_model,                 # функция создания модели
    objective='val_accuracy',    # метрика, которую нужно оптимизировать - 
                                 # доля правильных ответов на проверочном наборе данных
    max_trials=200,               # максимальное количество запусков обучения 
    executions_per_trial=1,
    max_model_size = 2040109465,
    directory='test_directory'   # каталог, куда сохраняются обученные сети  
    )

# tuner.search_space_summary()
# print ("Обучаем модель с использованием генераторов")
# tuner.search(train_dataset, epochs=epochs, validation_data=validation_dataset)


models = tuner.get_best_models(num_models=5)

for model in models:
    model.summary()
    model.evaluate(test_dataset)
    print('-------')

print (tuner.get_best_hyperparameters()[0].values)
print (tuner.results_summary())