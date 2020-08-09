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


#load fitched
features_train = np.load(open('features_train.npy', 'rb'))
features_val = np.load(open('features_val.npy', 'rb'))
features_test = np.load(open('features_test.npy', 'rb'))

#generate try answer
labels_train = np.array([0]*(nb_train_samples // 2)
        + [1]*(nb_train_samples // 2))

labels_val = np.array([0]*(nb_validation_samples // 2)
        + [1]*(nb_validation_samples // 2))

labels_test = np.array([0]*(nb_test_samples // 2)
        + [1]*(nb_test_samples // 2))

#Архитектура сети
model = Sequential()
model.add(Flatten(input_shape=features_train.shape[1:]))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))

#compilation net
model.compile(optimizer='Adam',
        loss='binary_crossentropy',
        metrics = ['accuracy'])

#learnig model
model.fit(features_train, labels_train,
        epochs=15,
        batch_size=64,
        validation_data=(features_val, labels_val),
        verbose=2)

scores = model.evaluate(features_test, labels_test,
        verbose=1)
print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))