from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from  tensorflow.python.keras.preprocessing.image import image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

#показываем картинку
def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.xlabel(img_path)
        plt.show()

    return img_tensor

def show_image(img, label='---'):
    plt.imshow(img[0])                           
    plt.axis('off')
    plt.title(label)
    plt.show()



def viewImage(image, window_name='------'):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#путь до модели
model_backup = 'save\\mnist-dense-26-0.9441.hdf5'

#каталог с картинками
image_dir = 'D:\\work\\ATM_foto\\image_test'
img_width, img_height = 150, 150
input_shape = (img_width, img_height, 3)


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

model.load_weights(model_backup)
print ('model is load...')

print ("Компилируем нейронную сеть")
model.compile(loss='binary_crossentropy',
              optimizer='adamax',
              metrics=['accuracy'])



#чтение картинок из директории
datagen = ImageDataGenerator(rescale=1. / 255)
for file_name in os.listdir(image_dir):
    image_new = load_image(image_dir+'\\'+file_name, 0)
    result = model.predict(image_new)
    print (file_name,'-->','{:0.10f}'.format(result[0][0]), '--> {_a}'.format(_a = 'Work' if result[0][0] <= 0.5 else 'Family'))
    show_image(image_new, file_name+' --> {_a}'.format(_a = 'Work' if result[0][0] <= 0.5 else 'Family'))
