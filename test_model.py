from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from keras.layers import Dropout, BatchNormalization, Input
from tensorflow import keras
from keras.models import Model
import tensorflow as tf

def model_v4():
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
    model_backup = 'save\\ver4_gpu-42-0.0104-0.9588-Test95.56.hdf5'
    print ("Загружаем веса модели из сохраненки",model_backup)
    model.load_weights(model_backup)
    return model

def model_v3():
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
    model_backup = 'save\\ver3_gpu-37-0.0335-0.9564-Test94.79.hdf5'
    print ("Загружаем веса модели из сохраненки",model_backup)
    model.load_weights(model_backup)
    return model

def model_v2():
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
    model_backup = 'save\\ver2__model.hdf5'
    print ("Загружаем веса модели из сохраненки",model_backup)
    model.load_weights(model_backup)
    return model

def model_v1():
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
    model_backup = 'save\\mnist-dense-stage3-03-0.9599.hdf5'
    print ("Загружаем веса модели из сохраненки",model_backup)
    model.load_weights(model_backup)
    return model

def model_v0():
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
    model_backup = 'save\\mnist-dense-no-image-03-0.9534.hdf5'
    print ("Загружаем веса модели из сохраненки",model_backup)
    model.load_weights(model_backup)
    return model

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
epochs = 50
# Размер мини-выборки
batch_size = 32#16
# Количество изображений для обучения
nb_train_samples = 90019
# Количество изображений для проверки
nb_validation_samples = 19290
# Количество изображений для тестирования
nb_test_samples = 19291

train_datagen = ImageDataGenerator(rescale=1. / 255)
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

for model, nmodel in zip([model_v0(), model_v1(), model_v2(), model_v3(), model_v4()], ['0', '1', '2', '3', '4']):
    scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
    print(f"Model V{nmodel}. "+"Аккуратность на тестовых данных: %.2f%%\n--------------" % (scores[1]*100))

# Model V0. Аккуратность на тестовых данных: 96.00%
# --------------
# Model V1. Аккуратность на тестовых данных: 94.26%
# --------------
# Model V2. Аккуратность на тестовых данных: 95.34%
# --------------
# Model V3. Аккуратность на тестовых данных: 96.17%
# --------------
# Model V4. Аккуратность на тестовых данных: 95.63%
