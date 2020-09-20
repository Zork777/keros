from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from  tensorflow.python.keras.preprocessing.image import image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import PySimpleGUI as sg
import exifread
import datetime


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
model_backup = 'save\\mnist-dense-stage3-03-0.9599.hdf5'

#каталог с картинками
#image_dir = 'D:\\work\\ATM_foto\\image_test'
image_dir = 'Z:\\мой телефон'
img_width, img_height = 150, 150
input_shape = (img_width, img_height, 3)

#create model
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
#datagen = ImageDataGenerator(rescale=1. / 255)
df = pd.DataFrame(columns=['file_name','flag'])
info_pict = {'file_name':'', 'result':0, 'flag':'', 'day_week':0, 'hour':0, 'datetime':''}

for i, file_name in enumerate(os.listdir(image_dir)):
#    if i> 20: break
    day_week = np.NaN
    hour = np.NaN
    dt = np.NaN

#read EXIF info in file
    try:
        img = open(image_dir+'\\'+file_name, 'rb')
    except Exception:
        print ('error load file for EXIF-',file_name)
        continue
    else:
        tags = exifread.process_file(img)
        try:
            dt = datetime.datetime.strptime(str(tags['EXIF DateTimeOriginal']), '%Y:%m:%d %H:%M:%S')
        except Exception:
            print ('file_name:{} ---->EXIF not found....'.format(file_name))
        else:    
            day_week = datetime.datetime.weekday(dt)+1
            hour = dt.hour
    img.close()

#read file for prediction class work or family
    try:
        image_new = load_image(image_dir+'\\'+file_name, 0)
    except Exception:
        print ('error load file for prediction-',file_name)
        continue

    result = model.predict(image_new)
    sg.one_line_progress_meter('progress meter', i+1, len(os.listdir(image_dir)), '-key-')
#    print ('{}'.format(i), file_name,'-->','{:0.10f}'.format(result[0][0]), '--> {_a}'.format(_a = 'Work' if result[0][0] <= 0.5 else 'Family'))
    info_pict['file_name']=file_name
    info_pict['result']=result[0][0]
    info_pict['flag']='{_a}'.format(_a = 'Work' if result[0][0] <= 0.99 else 'Family')
    info_pict['day_week']=day_week if np.isnan(day_week) else int(day_week)
    info_pict['hour']=hour if np.isnan(hour) else int(hour)
    info_pict['datetime']=dt
    df = df.append(info_pict, ignore_index=True)
#    show_image(image_new, file_name+' --> {_a}'.format(_a = 'Work' if result[0][0] <= 0.8 else 'Family')+'|{:0.10f}'.format(result[0][0]))
df.to_csv('list_image.csv', index_label='N')
