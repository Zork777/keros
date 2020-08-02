from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
import os

#показываем картинку
def viewImage(image, window_name='------'):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#путь до модели
model_dir = 'save\\mnist-dense-07-0.9397.hdf5'

#каталог с картинками
image_dir = 'D:\\work\\ATM_foto\\image_test'

input_shape = (img_width, img_height, 3)


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

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

loaded_model.load_weights()

print ("Компилируем нейронную сеть")
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



#чтение картинок из директории
for file in os.listdir(image_dir):

    image = cv2.imread(image_dir+'\\'file_name)

    viewImage(image, file)
    print (file)
