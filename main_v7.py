import tensorflow
#ImageDataGenerator = tensorflow.keras.preprocessing.image.ImageDataGenerator
Adam = tensorflow.keras.optimizers.Adam
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


from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential, Model
# from keras.layers import Conv2D, MaxPooling2D
# from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
# from keras.optimizers import SGD, Adam
# from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

import matplotlib.pyplot as plt
import os
import datetime

#from keras.models import model_from_json

# Каталог с данными для обучения
train_dir = 'C:\\work1\\ATM_foto\\train'
# Каталог с данными для проверки
val_dir = 'C:\\work1\\ATM_foto\\val'
# Каталог с данными для тестирования
test_dir = 'C:\\work1\\ATM_foto\\test'
# Размеры изображения

img_width, img_height = 110, 110

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

# Сохраняем сеть на каждой эпохе
# {epoch:02d} - номер эпохи
# {val_acc:.4f} - значение аккуратности на проверочном ноборе данных
# callbacks = [ModelCheckpoint('save/mnist-dense-{epoch:02d}-{val_acc:.4f}.hdf5')]
# Сохраняем только лучший вариант сети
# загружаем веса из сохраненки



log_dir = 'logs_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint = ModelCheckpoint('save_big_test\\ver7_ver2-{epoch:02d}-{val_accuracy:.4f}.hdf5', monitor='val_accuracy', save_best_only=True)
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

print ("Обучаем модель с использованием генераторов")
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[checkpoint, tensorboard_callback])

scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

#Аккуратность на тестовых данных: 90.4%