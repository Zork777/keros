from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from keras.models import Model
import os


#from keras.models import model_from_json

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

#Архитектура сети
#Слой свертки, размер ядра 3х3, количество карт признаков - 32 шт., функция активации ReLU.
#Слой подвыборки, выбор максимального значения из квадрата 2х2
#Слой свертки, размер ядра 3х3, количество карт признаков - 32 шт., функция активации ReLU.
#Слой подвыборки, выбор максимального значения из квадрата 2х2
#Слой свертки, размер ядра 3х3, количество карт признаков - 64 шт., функция активации ReLU.
#Слой подвыборки, выбор максимального значения из квадрата 2х2
#Слой преобразования из двумерного в одномерное представление
#Полносвязный слой, 64 нейрона, функция активации ReLU.
#Слой Dropout.
#Выходной слой, 1 нейрон, функция активации sigmoid
#Слои с 1 по 6 используются для выделения важных признаков в изображении, а слои с 7 по 10 - для классификации.

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

# checkpoint = ModelCheckpoint('save_big_test\\ver3_T_ver_best_2-{epoch:02d}-{val_accuracy:.4f}.hdf5', monitor='val_accuracy', save_best_only=True)
# print ("Обучаем модель с использованием генераторов")
# history = model.fit_generator(
#         train_generator,
#         steps_per_epoch=nb_train_samples // batch_size,
#         epochs=epochs,
#         validation_data=val_generator,
#         validation_steps=nb_validation_samples // batch_size,
#         callbacks=[checkpoint])

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')


scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))
plt.show()
#90,69%