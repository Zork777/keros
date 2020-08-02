from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
#from keras.models import model_from_json

# Каталог с данными для обучения
train_dir = 'D:\\work\\ATM_foto\\train'
# Каталог с данными для проверки
val_dir = 'D:\\work\\ATM_foto\\val'
# Каталог с данными для тестирования
test_dir = 'D:\\work\\ATM_foto\\test'
# Размеры изображения
img_width, img_height = 250, 250
# Размерность тензора на основе изображения для входных данных в нейронную сеть
# backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)
# Количество эпох
epochs = 30
# Размер мини-выборки
batch_size = 16
# Количество изображений для обучения
nb_train_samples = 4291
# Количество изображений для проверки
nb_validation_samples = 920
# Количество изображений для тестирования
nb_test_samples = 920

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
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(128)) #64
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

print ("Компилируем нейронную сеть")
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

datagen = ImageDataGenerator(rescale=1. / 255)

print ("Генератор данных для обучения на основе изображений из каталога")
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

print ("Генератор данных для проверки на основе изображений из каталога")
val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

print ("Генератор данных для тестирования на основе изображений из каталога")
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# Сохраняем сеть на каждой эпохе
# {epoch:02d} - номер эпохи
# {val_acc:.4f} - значение аккуратности на проверочном ноборе данных
# callbacks = [ModelCheckpoint('save/mnist-dense-{epoch:02d}-{val_acc:.4f}.hdf5')]
# Сохраняем только лучший вариант сети
callbacks = [ModelCheckpoint('save\\mnist-dense-{epoch:02d}-{val_acc:.4f}.hdf5', monitor='val_loss', save_best_only=True)]

print ("Обучаем модель с использованием генераторов")
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=callbacks)

scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))

