from keras.preprocessing.image import ImageDataGenerator
import os

def generate_image(image_dir, target_dir, n_images):
    img_width, img_height = 224, 224
    datagen = ImageDataGenerator( 
        rescale=1. / 255,
        rotation_range=10, 
        width_shift_range=0.15, 
        height_shift_range=0.15,
        zoom_range=0.2,
        shear_range=0.15,
    #    brightness_range=[0.5, 2], 
        horizontal_flip=True,
        fill_mode='nearest')

    n=0
    n_files = len (os.listdir(image_dir+'\\foto'))
    for batch in datagen.flow_from_directory(
                            image_dir,
                            target_size=(img_width, img_height),
                            batch_size=n_images, #кол-во генерируемых фотографий, отчет от 0
                            class_mode='binary',
                            shuffle=False,
                            save_to_dir=target_dir,
                            save_prefix='hi',
                            save_format='jpg'):
        n+=1
        if n > n_files: break
        print (n,'-->', 'осталось-',n_files-n,'\r', end='')

# Каталог с данными для обучения
#work
# image_dir = 'D:\\work\\ATM_foto\\Source_telefone\\work_foto'
# target_dir = 'C:\\work\\ATM_foto_source\\atm'
# generate_image (image_dir, target_dir, 13)

#home
image_dir = 'D:\\work\\ATM_foto\\Source_telefone\\home_foto'
target_dir = 'C:\\work\\ATM_foto_source\\other'
generate_image (image_dir, target_dir, 13)


