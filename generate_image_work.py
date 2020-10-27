from keras.preprocessing.image import ImageDataGenerator
import os


# Каталог с данными для обучения
image_dir = 'D:\\work\\ATM_foto\\Source_telefone\\work_foto'
target_dir = 'C:\\work\\ATM_foto\\work'
img_width, img_height = 150, 150

datagen = ImageDataGenerator( 
    rescale=1. / 255,
    rotation_range=10, 
    width_shift_range=0.15, 
    height_shift_range=0.15,
    zoom_range=0.2,
    shear_range=0.15,
    brightness_range=[0.5, 2], 
    horizontal_flip=True,
    fill_mode='nearest')

n=0
n_files = len (os.listdir(image_dir+'\\work'))
for batch in datagen.flow_from_directory(
                        image_dir,
                        target_size=(img_width, img_height),
                        batch_size=13, #кол-во генерируемых фотографий, отчет от 0
                        class_mode='binary',
                        shuffle=False,
                        save_to_dir=target_dir,
                        save_prefix='hi',
                        save_format='jpg'):
    n+=1
    if n > n_files: break
    print (n,'-->', 'осталось-',n_files-n,'\r', end='')