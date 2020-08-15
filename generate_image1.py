from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

# Каталог с данными для обучения
image_dir = 'D:\\work\\ATM_foto\\Source\\other'

def generate_image(img, name_file):
    for i,b in zip(range(20), datagen.flow(
                    x,
                    batch_size=1,
                    save_to_dir=image_dir,
                    save_prefix='gen_'+name_file,
                    save_format='jpg')):
        pass
#        print (i,'\r')


datagen = ImageDataGenerator( 
    rotation_range=40, 
    width_shift_range=0.2, 
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

#чтение картинок из директории
n=0
n_files = len (os.listdir(image_dir))
for file_name in os.listdir(image_dir):
    n+=1
    image_ = load_img(image_dir+'\\'+file_name)
    x = img_to_array(image_)
    x = x.reshape((1,) + x.shape)
    print (file_name,'-->', 'осталось-',n_files-n,'\r', end='')
    generate_image(x, file_name)
