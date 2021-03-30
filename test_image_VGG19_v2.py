import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, shutil
import datetime, time
from  tensorflow.python.keras.preprocessing.image import image
from tensorflow import keras
from keras.models import load_model
import argparse
from sort_foto import sort_foto
import PySimpleGUI as sg


#создаем парсер команд
parser = argparse.ArgumentParser(description='Разделение фотографий на рабочие и домашние. (zork777)',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('path', help='путь где лежат фотографии')
parser.add_argument('--hour', type=int, default=7, help='во сколько часов сканировать папку с фотографиями')
parser.add_argument('--test', help='тестовый режим, без перемещения файлов')
args = parser.parse_args()

#определяем переменные
image_dir = vars(args)['path']
timer_hour = vars(args)['hour']
test_mode = vars(args)['test']
img_width, img_height = 224, 224
input_shape = (img_width, img_height, 3)
image_dir_target_work = 'work_foto'
image_dir_target_home = 'home_foto'


if test_mode:
    print ('*****test mode ON*****. Press ctrl^c for exit programm.')
else:
    print ('*****test mode OFF*****')


class Work_file:
    def __init__(self, path, dir_work, dir_home):
        self.path = path
        self.image_dir_target_work = path + '\\' + dir_work
        self.image_dir_target_home = path + '\\' + dir_home
        if self.create_dir():
            exit(1)

    def create_dir(self):
        if os.path.isdir(self.image_dir_target_work):
            print (f'Директория найдена: {self.image_dir_target_work}')
        else:
            try:
                print (f'Создаем директорию: {self.image_dir_target_work}')
                os.mkdir(self.image_dir_target_work)
            except Exception:
                print (f'Не удалось создать директорию: {self.image_dir_target_work}')
                return True
        
        
        if  os.path.isdir(self.image_dir_target_home):
            print (f'Директория найдена: {self.image_dir_target_home}')            
        else:
            try:
                print (f'Создаем директорию: {self.image_dir_target_home}')
                os.mkdir(self.image_dir_target_home)
            except Exception:
                print (f'Не удалось создать директорию: {self.image_dir_target_home}')
                return True

    def move_file(self, file_name, flag):
        self.file_name = file_name
        self.flag = flag
        self.source_file = f'{self.path}\\{self.file_name}'
        self.target_file = '{_a}'.format(_a = self.image_dir_target_work if self.flag == 'Work' else self.image_dir_target_home)+f'\\{self.file_name}'
        print (f'{self.flag.upper()}-{self.source_file} ---> '+self.target_file)
        try:
            shutil.move(self.source_file, self.target_file)
        except Exception:
            print (f'ERROR-Не удалось переместить файл {self.source_file} в {self.target_file}')
        
        

#загружаем картинку
def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
    return img_tensor

#показываем картинку
def show_image (img_tensor, label='----'):
    plt.imshow(img_tensor[0])                           
    plt.axis('off')
    plt.title(label)
    plt.show()

class Model:
    """
    создаем модель
    """
    def __init__(self, model_filename):
        """
        конструктор
        """
        self.model_filename = model_filename
    
    def create(self):
        print (f'создаем модель на основе сохраненки:{self.model_filename}')
        model = load_model(self.model_filename)
        model.compile(loss='binary_crossentropy',
            optimizer=keras.optimizers.Adamax(lr=0.0001),
            metrics=['accuracy'])
        return model
    

def main():
    #create model
    model = Model('save_big_test\\main_VGG19_v2.h5')
    model = model.create()
    work_file = Work_file(image_dir, image_dir_target_work, image_dir_target_home)

    file_name_images = []
    #чтение картинок из директории
    print (f'Смотрим в каталог: {image_dir}')
    try:
        for _, file_name in enumerate(os.listdir(image_dir)):
            file_name_images.append(file_name)
    except Exception:
        print (f'Каталог {image_dir} не найден')
        exit


    #read file for prediction class work or family
    for i, file_name in enumerate(file_name_images):
        if test_mode:
            file_name = file_name_images[np.random.randint(0, len(file_name_images), 1)[0]]
        else:
            sg.one_line_progress_meter('progress meter', i+1, len(file_name_images), '-key-') #progress indicator

        try:
            img = load_image(image_dir+'\\'+file_name)
        except Exception:
            print ('error load file for prediction-',file_name)
            continue
        result = model.predict(img)
        if test_mode:
            show_image(img, '{_a}'.format(_a = 'Рабочее фото' if result[0][0] < 0.5 else 'Личное фото'))
        else:
            work_file.move_file(file_name, '{_a}'.format(_a = 'Work' if result[0][0] < 0.5 else 'Family'))


if __name__ == '__main__':
    while 1:
        date_now = datetime.datetime.now()
        print ('Сканирование фото в директории произойдет в {} часов- сейчас: {}\r'.format(timer_hour, datetime.datetime.now().isoformat()), end='')
        if date_now.hour == (timer_hour if not test_mode else date_now.hour):
            main()
            print('Сканирование директории завершено.')
            print('Старт сортировки фото')
            sort_foto(image_dir+image_dir_target_home)
            sort_foto(image_dir+image_dir_target_work)
        time.sleep(60*30)