from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import matplotlib.pyplot as plt

# Каталог с данными для обучения
image_dir = 'D:\\work\\ATM_foto\\Source\\other'
img_width, img_height = 150, 150

def generate_image(img, name_file):
    for i,b in zip(range(1), datagen.flow(
                    img,
                    batch_size=1,
                    save_to_dir=image_dir,
                    save_prefix='gen_'+name_file,
                    save_format='jpg',
                    target_size=(img_width, img_height))):
        pass
#        print (i,'\r')


datagen = ImageDataGenerator(rescale=1. / 255) 

# datagen = ImageDataGenerator( 
#     rotation_range=40, 
#     width_shift_range=0.2, 
#     height_shift_range=0.2,
#     zoom_range=0.2,
#     shear_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest')
n=0
n_files = len (os.listdir(image_dir+'\\other'))
for batch in datagen.flow_from_directory(
                        image_dir,
                        target_size=(img_width, img_height),
                        batch_size=1,
                        class_mode='binary',
                        shuffle=False,
                        save_to_dir=image_dir+'img',
                        save_prefix='hi',
                        save_format='jpg'):
    n+=1
    if n > n_files: break
    print (n,'-->', 'осталось-',n_files-n,'\r', end='')

# for n, image in zip(range(6), test_generator):
#     plt.imshow(image) #Needs to be in row,col order
#     plt.savefig(image_dir+'\\'+str(n)+'_im.jpg')

#чтение картинок из директории
# n=0
# n_files = len (os.listdir(image_dir))
# for file_name in os.listdir(image_dir):
#     n+=1
#     image_ = load_img(image_dir+'\\'+file_name)
#     x = img_to_array(image_)
#     x = x.reshape((1,) + x.shape)
#     print (file_name,'-->', 'осталось-',n_files-n,'\r', end='')
#     generate_image(x, file_name)
