from imutils import paths
from glob import glob
import pandas as pd
import os
import cv2
import numpy as np

d_path = '/home/faysal/PycharmProjects/DiabeticRetnopathy/TrainingPipeline/DR_Training_pipeline/data/Aptos_Eyepacs_USydney_224/Aptos/224/'
csv_path = '/home/faysal/PycharmProjects/DiabeticRetnopathy/TrainingPipeline/DR_Training_pipeline/data/Aptos_Eyepacs_USydney_224/Aptos/train.csv'

save_path = '/home/faysal/PycharmProjects/DiabeticRetnopathy/TrainingPipeline/DR_Training_pipeline/data/warmup/' 

df=pd.read_csv(csv_path)
img_data__path=glob(d_path+'/*')


def prep(im_path,df,img_name_list,img_lbl_list,folder_name):
	img = cv2.imread(i)
	image_name = (i.split(os.path.sep)[-1]).split(".")[0]
	lb = df.loc[df.id_code == image_name, 'diagnosis'].values
	lb = lb.item()
	print(lb)
	print(type(lb))
	image_name = image_name+'.jpeg'
	img_name_list.append(image_name)
	img_lbl_list.append(lb)
	cv2.imwrite(str(save_path)+str(folder_name)+'/'+str(image_name),np.array(img))

	return img_name_list,img_lbl_list

img_name_list_train = []
img_lbl_list_train = []

img_name_list_val = []
img_lbl_list_val = []

c=0
for i in img_data__path:
	if c<=42:
		img_name_list_train,img_lbl_list_train = prep(i,df,img_name_list_train,img_lbl_list_train,"train")
		c+=1
	
	elif c<70:
		img_name_list_val,img_lbl_list_val = prep(i,df,img_name_list_val,img_lbl_list_val,"validation")
		c+=1


print(img_lbl_list_train)

final_df_train = pd.DataFrame(
    {'id_code': img_name_list_train,
     'diagnosis': img_lbl_list_train
    })

final_df_val = pd.DataFrame(
    {'id_code': img_name_list_val,
     'diagnosis': img_lbl_list_val
    })



print(final_df_train.shape)
print(final_df_val.shape)


final_df_train.to_csv(save_path+'/train.csv')
final_df_val.to_csv(save_path+'/val.csv')