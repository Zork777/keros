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
image_dataset_from_directory = tensorflow.keras.preprocessing.image_dataset_from_directory


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

print ("Генератор данных для обучения на основе изображений из каталога")
train_dataset = image_dataset_from_directory(train_dir, batch_size=batch_size, image_size=image_size)
validation_dataset = image_dataset_from_directory(val_dir, batch_size=batch_size, image_size=image_size)
test_dataset = image_dataset_from_directory(test_dir, batch_size=batch_size, image_size=image_size)

AUTOTUNE = tensorflow.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


def create_model(hp):
    model = Sequential()
    
    model.add(Conv2D(hp.Int(f'count_conv2d_neur_layer-1', min_value=32, max_value=256, step=32), (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(hp.Float(f'dropout_Dense_layer-1', min_value=0.1, max_value=0.5, step=0.1)))

    model.add(Conv2D(hp.Int(f'count_conv2d_neur_layer-2', min_value=32, max_value=128, step=32), (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(hp.Float(f'dropout_Dense_layer-2', min_value=0.1, max_value=0.5, step=0.1)))

    model.add(Conv2D(hp.Int(f'count_conv2d_neur_layer-3', min_value=32, max_value=128, step=32), (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(hp.Float(f'dropout_Dense_layer-3', min_value=0.1, max_value=0.5, step=0.1)))

    model.add(Flatten())
    model.add(Dense(hp.Int(f'count_neur_Dense_layer-4', min_value=250, max_value=1000, step=250), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float(f'dropout_Dense_layer-4', min_value=0.1, max_value=0.5, step=0.1)))

    model.add(Dense(hp.Int(f'count_neur_Dense_layer-5', min_value=250, max_value=1000, step=250), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float(f'dropout_Dense_layer-5', min_value=0.1, max_value=0.5, step=0.1)))

    model.add(Dense(hp.Int(f'count_neur_Dense_layer-6', min_value=100, max_value=400, step=100), activation='relu'))
    model.add(Dense(hp.Int(f'count_neur_Dense_layer-7', min_value=100, max_value=400, step=100), activation='relu'))
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
tuner.search(train_dataset, epochs=epochs, validation_data=validation_dataset)


models = tuner.get_best_models(num_models=5)

for model in models:
    model.summary()
    model.evaluate(test_dataset)
    print('-------')

print (tuner.get_best_hyperparameters()[0].values)
print (tuner.results_summary())