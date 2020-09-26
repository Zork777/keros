import sys
import pandas as pd
import shutil
import PySimpleGUI as sg

image_dir_source = 'Z:\\мой телефон'
image_dir_target_work = 'Z:\\work_foto'
image_dir_target_home = 'Z:\\home_foto'

if len (sys.argv) > 1:
    print ('filter by:',sys.argv[1])
    filter_result = float(sys.argv[1])
    #df = pd.read_csv('list_image.csv', sep=';', header=0)
    df = pd.read_csv('del_list_merge.csv')
    _df = df[df['result']>filter_result].drop(columns=['N'])
    _df.to_csv('list_image_work.csv', index_label='N')
    len_df = len(df)
    for n, file_name, result in zip(range(len_df), df['file_name'], df['result']):
        sg.one_line_progress_meter('progress meter', n, len_df, '-key-')
        image_dir_target = image_dir_target_home if result > filter_result else image_dir_target_work
        try:
            shutil.copy2(image_dir_source+'\\'+file_name, image_dir_target+'\\'+str(round(result, 3))+'_'+file_name)
        except Exception:
            print ('{n}: {file_name}---->error: file skip'.format(n=n, file_name=file_name))
            continue
        else:
            print ('{n}: {file_name}---->move file ({result})'.format(n=n, result = result, file_name=image_dir_target+'\\'+file_name))
else:
    print ('enter float number')