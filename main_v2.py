import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from keras.models import Model
from keras.layers import Dense, Flatten, Input
from keras.layers import Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


classes=['work', 'family']
# Каталог с данными для обучения
train_dir = 'D:\\work\\ATM_foto\\train'
# Каталог с данными для проверки
val_dir = 'D:\\work\\ATM_foto\\val'
# Каталог с данными для тестирования
test_dir = 'D:\\work\\ATM_foto\\test'
# Размеры изображения
img_width, img_height = 150, 150
# Размерность тензора на основе изображения для входных данных в нейронную сеть
# backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)
# Количество эпох
epochs = 50
# Размер мини-выборки
batch_size = 32#16
# Количество изображений для обучения
nb_train_samples = 90019
# Количество изображений для проверки
nb_validation_samples = 19290
# Количество изображений для тестирования
nb_test_samples = 19291


def create_model():
    input_layer = Input(shape=input_shape, dtype=tf.float32, name='Input')
    x = BatchNormalization()(input_layer)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(1, activation='softmax')(x)
    model = Model(inputs=[input_layer], outputs=[output_layer])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['sparse_categorical_accuracy'])
    return model   

cpu_model = create_model()

train_datagen = ImageDataGenerator(rescale=1. / 255, 
    rotation_range=40, 
    width_shift_range=0.2, 
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)

print ("Генератор данных для обучения на основе изображений из каталога")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

print ("Генератор данных для проверки на основе изображений из каталога")
val_generator = test_datagen.flow_from_directory(
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

# Сохраняем сеть на каждой эпохе
# {epoch:02d} - номер эпохи
# {val_acc:.4f} - значение аккуратности на проверочном ноборе данных
# callbacks = [ModelCheckpoint('save/mnist-dense-{epoch:02d}-{val_acc:.4f}.hdf5')]
# Сохраняем только лучший вариант сети
# загружаем веса из сохраненки

# model_backup = 'save\\mnist-dense-23-0.9530.hdf5'
# print ("Загружаем веса модели из сохраненки",model_backup)
# model.load_weights(model_backup)

callbacks = [ModelCheckpoint('save\\ver2-{epoch:02d}-{val_acc:.4f}.hdf5', monitor='val_loss', save_best_only=False)]

print ("Обучаем модель с использованием генераторов")

history = cpu_model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=callbacks)



# Оцениваем качество обучения модели на тестовых данных
scores = cpu_model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))

