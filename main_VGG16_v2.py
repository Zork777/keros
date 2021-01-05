import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import Model
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import matplotlib.pyplot as plt


# Каталог с данными для обучения
train_dir = 'C:\\work1\\ATM_foto\\train'
# Каталог с данными для проверки
val_dir = 'C:\\work1\\ATM_foto\\val'
# Каталог с данными для тестирования
test_dir = 'C:\\work1\\ATM_foto\\test'
# Размеры изображения
img_width, img_height = 224, 224
# Размерность тензора на основе изображения для входных данных в нейронную сеть
# backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)
# Количество эпох
epochs = 100
# Размер мини-выборки
batch_size = 16
# Количество изображений для обучения
nb_train_samples = len(os.listdir(train_dir+'\\atm'))
# Количество изображений для проверки
nb_validation_samples = len(os.listdir(val_dir+'\\atm'))
# Количество изображений для тестирования
nb_test_samples = len(os.listdir(test_dir+'\\atm'))

model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)

# mark loaded layers as not trainable
for layer in model.layers:
	layer.trainable = False

layer_dict = dict([(layer.name, layer) for layer in model.layers])

# define new model
x = layer_dict['block5_pool'].output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=model.input, outputs=output)

model.summary()

print ("Компилируем нейронную сеть model V3T")
model.compile(loss='binary_crossentropy',
            optimizer=keras.optimizers.Adam(lr=0.0001),#lr=0.0023599656),
            metrics=['accuracy'])

results=[]
result = {'file_name':'', 'result':0}

for file_name in os.listdir('save_big_test'):
    if 'VGG16-v2' in file_name:
        result['file_name'] = file_name
        model_backup = 'save_big_test\\'+file_name
        print ("Загружаем веса модели из сохраненки",model_backup)
        model.load_weights(model_backup)

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

# checkpoint = [ModelCheckpoint('save_big_test\\VGG16-v2-{epoch:02d}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5',
#                             monitor='val_accuracy',
#                             save_best_only=False),
#             EarlyStopping(monitor="loss", patience=5)]


# print ("Обучаем модель с использованием генераторов")
# history = model.fit_generator(
#         train_generator,
#         steps_per_epoch=nb_train_samples // batch_size,
#         epochs=epochs,
#         validation_data=val_generator,
#         validation_steps=nb_validation_samples // batch_size,
#         callbacks=checkpoint)

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')


        scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
        print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))
        result['result'] = scores[1]*100
        results.append(result.copy())

# plt.show()
for result in results:
    print('{file_name}-{result}'.format(file_name=result['file_name'], result=result['result']))

# VGG16-v2-06-0.4573-0.9389.hdf5-91.9432520866394
# VGG16-v2-11-0.0052-0.9331.hdf5-91.92987084388733
# VGG16-v2-37-0.0015-0.9412.hdf5-91.9432520866394
# VGG16-v2-38-0.0002-0.9371.hdf5-91.97002053260803
# VGG16-v2-46-0.0050-0.9394.hdf5-91.9432520866394