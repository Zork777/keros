import sys
import pandas as pd
import shutil
import PySimpleGUI as sg

image_dir_source = 'Z:\\мой телефон'
image_dir_target_work = 'Z:\\work_foto'
image_dir_target_home = 'Z:\\home_foto'
error_predict = '\\error_predict'

if len (sys.argv) > 1:
    print ('filter by:',sys.argv[1])
    filter_result = float(sys.argv[1])
    df = pd.read_csv('list_image.csv', header=0)
#    df = pd.read_csv('del_list_merge.csv')
    _df = df[df['result']>filter_result].drop(columns=['N'])
    _df.to_csv('list_image_work.csv', index_label='N')
    len_df = len(df)
    for n, file_name, result, day_week, hour in zip(range(len_df), df['file_name'], df['result'], df['day_week'], df['hour']):
#        print ('\n----------\nfile name:{}\nresult:{}\nday_week:{}\nhour:{}'.format(file_name, result, day_week, hour))
        error_predict = ''
#        if n > 100: break
        sg.one_line_progress_meter('progress meter', n, len_df, '-key-')

        flag = 'family' if result > filter_result else 'work'
        #проверяем условие когда work при этом в нераб. время
        if flag == 'work' and (day_week > 5 or (hour >= 18 or hour < 9)):
            flag = 'family' 
            error_predict = '\\error_predict'
        
        #проверяем условие когда family при этом в раб. время
        if flag == 'family' and (day_week < 6 or (hour < 18 and hour >= 9)):
            flag = 'work'
            error_predict = '\\error_predict'
        image_dir_target = image_dir_target_home+error_predict if flag == 'family' else image_dir_target_work+error_predict

        try:
#            print (image_dir_source+'\\'+file_name+'----->'+image_dir_target+'\\'+str(round(result, 3))+'_'+file_name)
            shutil.copy2(image_dir_source+'\\'+file_name, image_dir_target+'\\'+str(round(result, 3))+'_D'+str(day_week)+'_H'+str(hour)+'_'+file_name)
        except Exception:
            print ('{n}: {file_name}---->error: file skip'.format(n=n, file_name=file_name))
            continue
        else:
            print ('{n}: {file_name}---->move file ({result})'.format(n=n, result = result, file_name=image_dir_target+'\\'+file_name))
else:
    print ('enter float number')