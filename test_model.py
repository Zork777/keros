from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D
# from keras.layers import Dropout, Flatten, Dense
# from keras.callbacks import ModelCheckpoint
# from keras.callbacks import EarlyStopping
# import matplotlib.pyplot as plt
# from keras.layers import Dropout, BatchNormalization, Input
from tensorflow import keras
# from keras.models import Model
import tensorflow as tf
import os
import tensorflow
import numpy as np

Adam = tensorflow.keras.optimizers.Adam
Activation = tensorflow.keras.layers.Activation
Input = tensorflow.keras.layers.Input
Sequential = tensorflow.keras.Sequential
Model = tensorflow.keras.Model
layers = tensorflow.keras.layers
BatchNormalization = tensorflow.keras.layers.BatchNormalization
Conv2D = tensorflow.keras.layers.Conv2D
MaxPooling2D = tensorflow.keras.layers.MaxPooling2D
Flatten = tensorflow.keras.layers.Flatten
Dropout = tensorflow.keras.layers.Dropout
Dense = tensorflow.keras.layers.Dense
TensorBoard = tensorflow.keras.callbacks.TensorBoard
ModelCheckpoint = tensorflow.keras.callbacks.ModelCheckpoint
RandomNormal = tensorflow.keras.initializers.RandomNormal

def model_v10(train_generator, val_generator, test_generator):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=RandomNormal(stddev=np.sqrt(1/(img_height * img_width))),
                input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=RandomNormal(stddev=np.sqrt(2/(3*3*32)))))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer=RandomNormal(stddev=np.sqrt(2/(3*3*64)))))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer=RandomNormal(stddev=np.sqrt(2/(3*3*128)))))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(500, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(125, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    print ("Компилируем нейронную сеть")
    opt = Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
    model_backup = 'save_big_test\\ver10-98-0.9315.hdf5'
    print ("Загружаем веса модели из сохраненки",model_backup)
    model.load_weights(model_backup)
    return model

    global nb_train_samples, batch_size, nb_validation_samples, epochs
    checkpoint = ModelCheckpoint('save_big_test\\ver10-{epoch:02d}-{val_accuracy:.4f}.hdf5', monitor='val_accuracy', save_best_only=True)
    print ("Обучаем модель с использованием генераторов")
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[checkpoint])
    return model

def model_v9(train_generator, val_generator, test_generator):
    kernel_initializer = 'glorot_uniform' #tf.contrib.layers.xavier_initializer(uniform=False)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(500, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(250, activation='relu'))
    model.add(Dense(125, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))


    print ("Компилируем нейронную сеть")
    #opt = SGD(lr=0.001, momentum=0.9)
    opt = Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
    model_backup = 'save_big_test\\ver9-32-0.9400.hdf5'
    print ("Загружаем веса модели из сохраненки",model_backup)
    model.load_weights(model_backup)
    return model

    global nb_train_samples, batch_size, nb_validation_samples, epochs
    checkpoint = ModelCheckpoint('save_big_test\\ver9-{epoch:02d}-{val_accuracy:.4f}.hdf5', monitor='val_accuracy', save_best_only=True)
    print ("Обучаем модель с использованием генераторов")
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[checkpoint])

    return model

def model_v8(train_generator, val_generator, test_generator):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(500, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(250, activation='relu'))
    model.add(Dense(125, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))


    print ("Компилируем нейронную сеть ver 8")
    #opt = SGD(lr=0.001, momentum=0.9)
    opt = Adam(learning_rate=0.00001)
    model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
    model_backup = 'save_big_test\\ver8-18-0.8338.hdf5'
    print ("Загружаем веса модели из сохраненки",model_backup)
    model.load_weights(model_backup)

    global nb_train_samples, batch_size, nb_validation_samples, epochs
    checkpoint = ModelCheckpoint('save_big_test\\ver8-{epoch:02d}-{val_accuracy:.4f}.hdf5', monitor='val_accuracy', save_best_only=True)
    print ("Обучаем модель с использованием генераторов")
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=10,
        validation_data=val_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[checkpoint])
    
    return model

def model_v7(train_generator, val_generator, test_generator):
    img_width, img_height = 110, 110
    input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(1000, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(500, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(250, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))


    print ("Компилируем нейронную сеть ver7")
    opt = Adam(learning_rate=0.000112202)
    model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
    model_backup = 'save_big_test\\ver7-92-0.9359_110x110.hdf5'
    print ("Загружаем веса модели из сохраненки",model_backup)
    model.load_weights(model_backup)
        
    global nb_train_samples, batch_size, nb_validation_samples, epochs

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

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
    global nb_test_samples
    scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
    return scores


    checkpoint = ModelCheckpoint('save_big_test\\ver7-{epoch:02d}-{val_accuracy:.4f}.hdf5', monitor='val_accuracy', save_best_only=True)
    print ("Обучаем модель с использованием генераторов")
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[checkpoint])

    return model

def model_v6(train_generator, val_generator, test_generator):
    img_width, img_height = 110, 110
    input_shape = (img_width, img_height, 3)
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(1000, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(500, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(250, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))


    print ("Компилируем нейронную сеть ver 6")
    opt = Adam(learning_rate=0.000112202)
    model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
    model_backup = 'save_big_test\\ver6-37-0.9298.hdf5'
    print ("Загружаем веса модели из сохраненки",model_backup)
    model.load_weights(model_backup)

    global nb_train_samples, batch_size, nb_validation_samples, epochs
    
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

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

    global nb_test_samples
    scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
    return scores


    checkpoint = ModelCheckpoint('save_big_test\\ver6-{epoch:02d}-{val_accuracy:.4f}.hdf5', monitor='val_accuracy', save_best_only=True)
    print ("Обучаем модель с использованием генераторов")
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=200, #epochs,
        validation_data=val_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[checkpoint])

    return model

def model_v5(train_generator, val_generator, test_generator):
    model = Sequential()
    model.add(Conv2D(96, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(64)) #128
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print ("Компилируем нейронную сеть ver 5")
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    model_backup = 'save_big_test\\ver5-21-0.9337.hdf5'
    print ("Загружаем веса модели из сохраненки",model_backup)
    model.load_weights(model_backup)
    global batch_size, nb_test_samples
    scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
    return scores


    global nb_train_samples, nb_validation_samples, epochs
    checkpoint = ModelCheckpoint('save_big_test\\ver5-{epoch:02d}-{val_accuracy:.4f}.hdf5', monitor='val_accuracy', save_best_only=True)
    print ("Обучаем модель с использованием генераторов")
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=1, #epochs,
        validation_data=val_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[checkpoint])

    return model

def model_v4(train_generator, val_generator, test_generator):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.375))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print ("Компилируем нейронную сеть model V4")
    model.compile(loss='binary_crossentropy',
                optimizer='adamax',
                metrics=['accuracy'])
    model_backup = 'save_big_test\\ver4_T-06-0.9380.hdf5'
    print ("Загружаем веса модели из сохраненки",model_backup)
    model.load_weights(model_backup)
    global batch_size, nb_test_samples
    scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
    return scores


    global nb_train_samples, nb_validation_samples, epochs
    checkpoint = ModelCheckpoint('save_big_test\\ver4-{epoch:02d}-{val_accuracy:.4f}.hdf5', monitor='val_accuracy', save_best_only=True)
    print ("Обучаем модель с использованием генераторов")
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=1, #epochs,
        validation_data=val_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[checkpoint])

    return model

def model_v4_T(train_generator, val_generator, test_generator):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.375))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print ("Компилируем нейронную сеть model V4T")
    model.compile(loss='binary_crossentropy',
                optimizer='adamax',
                metrics=['accuracy'])
    model_backup = 'save_big_test\\ver4-27-0.9415.hdf5'
    print ("Загружаем веса модели из сохраненки",model_backup)
    model.load_weights(model_backup)
    global batch_size, nb_test_samples
    scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
    return scores


    global nb_train_samples, nb_validation_samples, epochs
    checkpoint = ModelCheckpoint('save_big_test\\ver4_T-{epoch:02d}-{val_accuracy:.4f}.hdf5', monitor='val_accuracy', save_best_only=True)
    print ("Обучаем модель с использованием генераторов")
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=1, #epochs,
        validation_data=val_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[checkpoint])

    return model

def model_v3_T(train_generator, val_generator, test_generator):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.125))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.5))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.5))


    model.add(Flatten())
    model.add(Dense(192)) #128
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print ("Компилируем нейронную сеть model V3T")
    model.compile(loss='binary_crossentropy',
                optimizer='adamax', 
                metrics=['accuracy'])
    model_backup = 'save_big_test\\ver3_T-01-0.9330.hdf5'
    print ("Загружаем веса модели из сохраненки",model_backup)
    model.load_weights(model_backup)
    global batch_size, nb_test_samples
    scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
    return scores


    global nb_train_samples, nb_validation_samples, epochs
    checkpoint = ModelCheckpoint('save_big_test\\ver3_T-{epoch:02d}-{val_accuracy:.4f}.hdf5', monitor='val_accuracy', save_best_only=True)
    print ("Обучаем модель с использованием генераторов")
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=1, #epochs,
        validation_data=val_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[checkpoint])

    return model

def model_v3(train_generator, val_generator, test_generator):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.125))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.5))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.5))


    model.add(Flatten())
    model.add(Dense(192)) #128
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print ("Компилируем нейронную сеть model V3")
    model.compile(loss='binary_crossentropy',
                optimizer='adamax', 
                metrics=['accuracy'])
    model_backup = 'save_big_test\\ver3-08-0.9413.hdf5'
    print ("Загружаем веса модели из сохраненки",model_backup)
    model.load_weights(model_backup)
    global batch_size, nb_test_samples
    scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
    return scores


    global nb_train_samples, nb_validation_samples, epochs
    checkpoint = ModelCheckpoint('save_big_test\\ver3-{epoch:02d}-{val_accuracy:.4f}.hdf5', monitor='val_accuracy', save_best_only=True)
    print ("Обучаем модель с использованием генераторов")
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=1, #epochs,
        validation_data=val_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[checkpoint])

    return model

def model_v2(train_generator, val_generator, test_generator):
    input_layer = Input(shape=input_shape, dtype=tf.float32, name='Input')
    x = BatchNormalization()(input_layer)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[input_layer], outputs=[output_layer])
    
    print ("Компилируем нейронную сеть model V2")
    model.compile(
        optimizer=keras.optimizers.Adamax(),#learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    model_backup = 'save_big_test\\ver2-68-0.9417.hdf5'
    print ("Загружаем веса модели из сохраненки",model_backup)
    model.load_weights(model_backup)
    global batch_size, nb_test_samples
    scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
    return scores


    global nb_train_samples, nb_validation_samples, epochs
    checkpoint = ModelCheckpoint('save_big_test\\ver2-{epoch:02d}-{val_accuracy:.4f}.hdf5', monitor='val_accuracy', save_best_only=True)
    print ("Обучаем модель с использованием генераторов")
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=1, #epochs,
        validation_data=val_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[checkpoint])

    return model

def model_v1(train_generator, val_generator, test_generator):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3))) #128
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64)) #128
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print ("Компилируем нейронную сеть model V1")
    model.compile(loss='binary_crossentropy',
                optimizer='adamax',
                metrics=['accuracy'])
    model_backup = 'save_big_test\\ver1-13-0.9311.hdf5'
    print ("Загружаем веса модели из сохраненки",model_backup)
    model.load_weights(model_backup)
    global batch_size, nb_test_samples
    scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
    return scores


    global nb_train_samples, nb_validation_samples, epochs
    checkpoint = ModelCheckpoint('save_big_test\\ver1-{epoch:02d}-{val_accuracy:.4f}.hdf5', monitor='val_accuracy', save_best_only=True)
    print ("Обучаем модель с использованием генераторов")
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=1, #epochs,
        validation_data=val_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[checkpoint])

    return model

def model_v0(train_generator, val_generator, test_generator):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(64, (3, 3))) #128
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Flatten())
    model.add(Dense(64)) #128
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print ("Компилируем нейронную сеть model V0")
    model.compile(loss='binary_crossentropy',
                optimizer='adamax', #adamax
                metrics=['accuracy'])
    model_backup = 'save_big_test\\ver0-89-0.9299.hdf5'
    print ("Загружаем веса модели из сохраненки",model_backup)
    model.load_weights(model_backup)
    global batch_size, nb_test_samples
    scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
    return scores

    global nb_train_samples, nb_validation_samples, epochs
    checkpoint = ModelCheckpoint('save_big_test\\ver0-{epoch:02d}-{val_accuracy:.4f}.hdf5', monitor='val_accuracy', save_best_only=True)
    print ("Обучаем модель с использованием генераторов")
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=1, #epochs,
        validation_data=val_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[checkpoint])

    return model

# Каталог с данными для обучения
train_dir = 'C:\\work1\\ATM_foto\\train'
# Каталог с данными для проверки
val_dir = 'C:\\work1\\ATM_foto\\val'
# Каталог с данными для тестирования
test_dir = 'C:\\work1\\ATM_foto\\test'
# Размеры изображения
img_width, img_height = 150, 150
# Размерность тензора на основе изображения для входных данных в нейронную сеть
# backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)
# Количество эпох
epochs = 100
# Размер мини-выборки
batch_size = 32#16
# Количество изображений для обучения
nb_train_samples = len(os.listdir(train_dir+'\\atm'))
# Количество изображений для проверки
nb_validation_samples = len(os.listdir(val_dir+'\\atm'))
# Количество изображений для тестирования
nb_test_samples = len(os.listdir(test_dir+'\\atm'))

train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

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

for scores, nmodel in zip([model_v0(train_generator, val_generator, test_generator),
                model_v1(train_generator, val_generator, test_generator),
                model_v2(train_generator, val_generator, test_generator),
                model_v3(train_generator, val_generator, test_generator),
                model_v4(train_generator, val_generator, test_generator),
                model_v3_T(train_generator, val_generator, test_generator),
                model_v4_T(train_generator, val_generator, test_generator),
                model_v5(train_generator, val_generator, test_generator),
                model_v6(train_generator, val_generator, test_generator),
                model_v7(train_generator, val_generator, test_generator)],
                ['0', '1', '2', '3', '4', '3T', '4T', '5', '6', '7']):
    print(f"Model V{nmodel}. "+"Аккуратность на тестовых данных: %.2f%%\n--------------" % (scores[1]*100))



# Model V0. Аккуратность на тестовых данных: 87.45%
# --------------
# Model V1. Аккуратность на тестовых данных: 88.16%
# --------------
# Model V2. Аккуратность на тестовых данных: 89.14%
# --------------
# Model V3. Аккуратность на тестовых данных: 88.51%
# --------------
# Model V4. Аккуратность на тестовых данных: 88.79%
# --------------
# Model V3T. Аккуратность на тестовых данных: 90.50%
# --------------
# Model V4T. Аккуратность на тестовых данных: 87.98%
# --------------
# Model V5. Аккуратность на тестовых данных: 88.89%
# --------------
# Model V6. Аккуратность на тестовых данных: 87.78%
# --------------
# Model V7. Аккуратность на тестовых данных: 90.36%
# --------------