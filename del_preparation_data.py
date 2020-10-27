import pandas as pd
import os
import exifread
import datetime
import matplotlib.pyplot as plt
import PySimpleGUI as sg


image_dir_target = 'Z:\\мой телефон'
df = pd.read_csv('list_merge.csv', usecols=['file_name','day_week',
                                            'hour','datetime','flag','result'])
#df = df.drop(['N'])
#print (df.head())
#print (df.info())
df = df.dropna()
#print (df.info())

df_ = df[df['flag']=='Work']# or df['hour']<9 or df['day_week']>5]

rows=8
columns=8
matrix=rows*columns

for n in range(int(len(df_)/matrix)+1):
    fig=plt.figure(figsize=(15, 15))
    fig.subplots_adjust(bottom=0.025, left=0.025, top = 0.9, right=0.9, wspace = 0.3, hspace = 0.1)
    df_bach = df_[n*matrix:n*matrix+matrix]
    for i, file_name, result in zip(range(1, matrix+1), df_bach['file_name'], df_bach['result']):
        img = plt.imread(image_dir_target+'\\'+file_name, 'rb')
        ax1= fig.add_subplot(rows, columns, i)
        ax1.axis('off')
        ax1.set_title(file_name+' '+str(result), fontsize=6)
        ax1.imshow(img)
    plt.tight_layout(True)
    plt.show()

# plt.Figure(figsize=(8,16))
# plt.xlabel('day week')
# plt.ylabel('count')
# plt.bar(df.groupby(by=['day_week']).count()['file_name'].index,df.groupby(by=['day_week']).count()['file_name'])
# plt.show()





