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

#classes=['work', 'family']
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
epochs = 3
# Размер мини-выборки
batch_size = 32#16
# Количество изображений для обучения
nb_train_samples = 90019
# Количество изображений для проверки
nb_validation_samples = 19290
# Количество изображений для тестирования
nb_test_samples = 19291

# train_datagen = ImageDataGenerator(rescale=1. / 255) 
# test_datagen = ImageDataGenerator(rescale=1. / 255)
# print(tf.__version__)
# print ("Генератор данных для обучения на основе изображений из каталога")
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='binary')

# print ("Генератор данных для проверки на основе изображений из каталога")
# val_generator = test_datagen.flow_from_directory(
#     val_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='binary')

# print ("Генератор данных для тестирования на основе изображений из каталога")
# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='binary')


def create_model(hp):
    model = Sequential()

    model.add(Conv2D(hp.Int('input_neur', min_value=32, max_value=64, step=32), (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for i in range(hp.Int('layers', 1, 5)):
        model.add(Conv2D(hp.Int(f'count_{i}_neur_layer', min_value=32, max_value=64, step=32), (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(hp.Float(f'dropout_{i}_layer', min_value=0.125, max_value=0.5, step=0.125)))

    model.add(Flatten())
    model.add(Dense(hp.Int('count_neur', min_value=64, max_value=256, step=64)))
    model.add(Activation('relu'))
    model.add(Dropout(hp.Float('dropout', min_value=0.25, max_value=0.5, step=0.25)))
    model.add(Dense(1, activation=hp.Choice('activation', values=['softmax', 'sigmoid'])))
    
    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam','adamax', 'rmsprop','SGD']),
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

# tuner.search_space_summary()
# print ("Обучаем модель с использованием генераторов")
# tuner.search(train_generator, epochs=3, validation_data=val_generator)


#models = tuner.get_best_models(num_models=5)

# for model in models:
#     model.summary()
#     model.evaluate_generator(test_generator, nb_test_samples // batch_size)
#     print('-------')

print (tuner.get_best_hyperparameters()[0].values)
print (tuner.results_summary())

# history = cpu_model.fit_generator(
#     train_generator,
#     steps_per_epoch=nb_train_samples // batch_size,
#     epochs=epochs,
#     validation_data=val_generator,
#     validation_steps=nb_validation_samples // batch_size,
#     callbacks=callbacks)



# # Оцениваем качество обучения модели на тестовых данных
# scores = cpu_model.evaluate_generator(test_generator, nb_test_samples // batch_size)
# print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# cpu_model.save_weights("save\\ver2__model.h5")

