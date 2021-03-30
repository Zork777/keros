import os, shutil
import datetime, time
import exifread

#image_dir_target_work = 'work_foto'
#image_dir_target_home = 'home_foto'
#image_dir = 'D:\\1\\'


def get_exif(file_name):
    my_image = open(file_name, 'rb')
    
    try:
        tag_date = exifread.process_file(my_image, details=False, stop_tag="DateTimeOriginal")['EXIF DateTimeOriginal']
    except Exception:
        tag_date = False
    
    my_image.close

    if tag_date:
        #found exif
        return datetime.datetime.strptime(tag_date.values, '%Y:%m:%d %H:%M:%S') #get date create foto
    else:
        #not found exif
        return datetime.datetime.fromtimestamp(os.path.getmtime(file_name))

def create_dir_date(path_dir_date):
    if os.path.isdir(path_dir_date):
        print (f'Директория найдена: {path_dir_date}')
    else:
        try:
            print (f'Создаем директорию: {path_dir_date}')
            os.mkdir(path_dir_date)
        except Exception:
            print (f'Не удалось создать директорию: {path_dir_date}')
            return True

def sort_foto(image_dir):
    #file_name_images = []
    #чтение картинок из директории
    print (f'Смотрим в каталог: {image_dir}')
    for _, file_name in enumerate(os.listdir(image_dir)):
#            file_name_images.append(file_name)
        try:
            file_date = get_exif(image_dir+"\\"+file_name)
            print (image_dir+"\\"+file_name+"--->", file_date.year, file_date.month)
            create_dir_date(image_dir+"\\"+str(file_date.year))
            create_dir_date(image_dir+"\\"+str(file_date.year)+"\\"+str(file_date.month))
            shutil.move(image_dir+"\\"+file_name, image_dir+"\\"+str(file_date.year)+"\\"+str(file_date.month)+"\\"+file_name) #move file
        except Exception:
            print (f'Ошибка при работе с файлом {image_dir}\\{file_name}')

#    return file_name_images


if __name__ == '__main__':
    sort_foto(image_dir+image_dir_target_home)
    sort_foto(image_dir+image_dir_target_work)