import numpy as np
import tensorflow as tf
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


# Каталог с данными для обучения
train_dir = 'C:\\work\\ATM_foto\\train'
# Каталог с данными для проверки
val_dir = 'C:\\work\\ATM_foto\\val'
# Каталог с данными для тестирования
test_dir = 'C:\\work\\ATM_foto\\test'
# Размеры изображения
img_width, img_height = 150, 150
# Размерность тензора на основе изображения для входных данных в нейронную сеть
# backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)
# Количество эпох
epochs = 2
# Размер мини-выборки
batch_size = 30#16

# Количество изображений для обучения
nb_train_samples = len(os.listdir(train_dir+'\\atm'))
# Количество изображений для проверки
nb_validation_samples = len(os.listdir(val_dir+'\\atm'))
# Количество изображений для тестирования
nb_test_samples = len(os.listdir(test_dir+'\\atm'))

train_datagen = ImageDataGenerator(rescale=1. / 255) 
test_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

print(tf.__version__)
print ("Генератор данных для обучения на основе изображений из каталога")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

print ("Генератор данных для проверки на основе изображений из каталога")
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

print ("Генератор данных для тестирования на основе изображений из каталога")
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


def create_model(hp):
    model = Sequential()

    model.add(Conv2D(hp.Int('input_neur', min_value=64, max_value=128, step=32), (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for i in range(1, hp.Int('layers_conv', 1, 3)):
        count_conv2d_size = hp.Int(f'count_conv2d_size_layer-{i}', min_value=3, max_value=5, step=1) 
        model.add(Conv2D(hp.Int(f'count_conv2d_neur_layer-{i}', min_value=32, max_value=64, step=32), (count_conv2d_size, count_conv2d_size))) #, padding='same'
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    
    for n in range(1, hp.Int('layers_conv', 1, 2)):
        model.add(Dense(hp.Int(f'count_neur_Dense_layer-{n}', min_value=250, max_value=1000, step=250)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float(f'dropout_Dense_layer-{n}', min_value=0.25, max_value=0.5, step=0.25)))

    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam','adamax']),
        loss='binary_crossentropy',
        metrics=['accuracy'])#sparse_categorical_accuracy'])
    return model   

tuner = RandomSearch(
    create_model,                 # функция создания модели
    objective='val_accuracy',    # метрика, которую нужно оптимизировать - 
                                 # доля правильных ответов на проверочном наборе данных
    max_trials=100,               # максимальное количество запусков обучения 
    executions_per_trial=2,
    directory='test_directory'   # каталог, куда сохраняются обученные сети  
    )

tuner.search_space_summary()
print ("Обучаем модель с использованием генераторов")
tuner.search(train_generator, epochs=epochs, validation_data=val_generator)


models = tuner.get_best_models(num_models=5)

for model in models:
    model.summary()
    model.evaluate_generator(test_generator, nb_test_samples // batch_size)
    print('-------')

print (tuner.get_best_hyperparameters()[0].values)
print (tuner.results_summary())