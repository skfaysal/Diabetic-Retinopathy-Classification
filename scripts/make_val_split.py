import os
import pandas as pd
from glob import glob
import cv2

# DEfine path
parent_dir =  os.path.normpath(os.getcwd() + os.sep + os.pardir)
data_path=parent_dir+'/data'+'/messidor/'

# read main train csv
train_df = pd.read_csv(str(data_path)+'messidorFull.csv')

# read main train iamges
train_images = glob(str(data_path)+'img_all'+'/*')

print(train_df.head(5))
print("shape of original dataframe:", train_df.shape)

# Creating a dataframe with 75%
# values of original dataframe
x_train = train_df.sample(frac = 0.75)

x_train.to_csv(data_path+'x_train.csv')
  
# Creating dataframe with 
# rest of the 25% values
x_test = train_df.drop(x_train.index)

x_test.to_csv(data_path+'x_test.csv')

print("\n75% of the givem DataFrame:")
print(x_train.shape)
  
print("\nrest 25% of the given DataFrame:")
print(x_test.shape)

# create train and validation folder

os.mkdir(str(data_path)+'train_images')
os.mkdir(str(data_path)+'validation_images')


for image in train_images:
	image_name_with_format = image.split(os.path.sep)[-1]

	image_name = image_name_with_format.split(".")[-2]
	img = cv2.imread(image)

	if image_name in x_train['image'].tolist():
		cv2.imwrite(str(data_path)+'train_images/'+str(image_name_with_format),img)
	else:
		cv2.imwrite(str(data_path)+'validation_images/'+str(image_name_with_format),img)

