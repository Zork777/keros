import shutil
import os

# Каталог с набором данных
data_dir = 'D:\\work\\ATM_foto\\source'
# Каталог с данными для обучения
train_dir = 'D:\\work\\ATM_foto\\train'
# Каталог с данными для проверки
val_dir = 'D:\\work\\ATM_foto\\val'
# Каталог с данными для тестирования
test_dir = 'D:\\work\\ATM_foto\\test'
# Часть набора данных для тестирования
test_data_portion = 0.15
# Часть набора данных для проверки
val_data_portion = 0.15
# Количество элементов данных в одном классе
nb_images = 6132

#Функция создания каталога с двумя подкаталогами по названию классов: 
def create_directory(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    os.makedirs(os.path.join(dir_name, "ATM"))
    os.makedirs(os.path.join(dir_name, "other"))

#Создание структуры каталогов для обучающего, проверочного и тестового набора данных
create_directory(train_dir)
create_directory(val_dir)
create_directory(test_dir)

#Функция копирования изображений в заданный каталог. Изображения котов и собак копируются в отдельные подкаталоги
def copy_images(start_index, end_index, source_dir, dest_dir):
    for i in range(start_index, end_index):
        shutil.copy2(os.path.join(source_dir, "ATM." + str(i) + ".jpg"), 
                    os.path.join(dest_dir, "ATM"))
        shutil.copy2(os.path.join(source_dir, "other." + str(i) + ".jpg"), 
                   os.path.join(dest_dir, "other"))

#Расчет индексов наборов данных для обучения, приверки и тестирования

start_val_data_idx = int(nb_images * (1 - val_data_portion - test_data_portion))
start_test_data_idx = int(nb_images * (1 - test_data_portion))
print(start_val_data_idx)
print(start_test_data_idx)

#Копирование изображений
copy_images(1, start_val_data_idx, data_dir, train_dir)
copy_images(start_val_data_idx, start_test_data_idx, data_dir, val_dir)
copy_images(start_test_data_idx, nb_images, data_dir, test_dir)
