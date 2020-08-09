from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np


# Каталог с данными для обучения
train_dir = 'D:\\work\\ATM_foto\\train'
# Каталог с данными для проверки
val_dir = 'D:\\work\\ATM_foto\\val'
# Каталог с данными для тестирования
test_dir = 'D:\\work\\ATM_foto\\test'
# Размеры изображения
img_width, img_height = 224, 224
# Размерность тензора на основе изображения для входных данных в нейронную сеть
# backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)
# Количество эпох
epochs = 30
# Размер мини-выборки
batch_size = 10#16
# Количество изображений для обучения
nb_train_samples = 4291
# Количество изображений для проверки
nb_validation_samples = 920
# Количество изображений для тестирования
nb_test_samples = 920


datagen = ImageDataGenerator(rescale=1. / 255)
print ("Генератор данных для обучения на основе изображений из каталога")
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

print ("Генератор данных для проверки на основе изображений из каталога")
val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)


print ("Генератор данных для тестирования на основе изображений из каталога")
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)


#Архитектура сети
#create model VGG16

vgg16_net = VGG16(weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3))

features_train = vgg16_net.predict_generator(
            train_generator,
            nb_train_samples // batch_size)
print ('features_train-',features_train.shape)

features_val = vgg16_net.predict_generator(
            val_generator,
            nb_validation_samples // batch_size)
print ('features_val-',features_val.shape)

features_test = vgg16_net.predict_generator(
            test_generator,
            nb_test_samples // batch_size)
print ('features_test-',features_test.shape)

#save fitch
print ('save fitch....')
np.save(open('features_train.npy', 'wb'), features_train)
np.save(open('features_val.npy', 'wb'), features_val)
np.save(open('features_test.npy', 'wb'), features_test)
print ('saved')
