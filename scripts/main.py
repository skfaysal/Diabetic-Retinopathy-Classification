import os
import pandas as pd
from glob import glob
# import custom modules
import prepareData
import DataFolder
import generators
import models

# Define data path
parent_dir =  os.path.normpath(os.getcwd() + os.sep + os.pardir)
data_path=parent_dir+'/data'

# define data path
Data_1 = data_path+'/Aptos_Eyepacs_USydney_224'+'/Aptos/train.csv'
Data_2 = data_path+'/Aptos_Eyepacs_USydney_224'+'/Eyepacs/train.csv'
Data_3 = data_path+'/Aptos_Eyepacs_USydney_224'+'/USydney/train.csv'

# define 5 fold csv path
fold_data = data_path+'/Aptos_Eyepacs_USydney_224'+'/5-fold.csv'

# Test data path
TEST_SET1 = data_path+'/resized-preprocess19/train19_images_ben_preprocessing_sigmaX10/train19_images_ben_preprocessing_sigmaX10/'
TEST_SET2 = data_path+'/resized-preprocessed15/train15_ben_preprocessing_sigmaX10/'

#Weight path
weight_path = parent_dir+'/weights/efficientnet-b5_imagenet_1000_notop.h5'

#output path
output_path = parent_dir+'/output/'

# # Parameters
# fold_num='fold_4'

# # Call preparedf method
# X_train,X_val,test,class_weights=prepareData.prepare_df(Data_1,Data_2,Data_3,fold_data,fold_num)

# # Call prepare data folder method
# train_dest_path,validation_dest_path = DataFolder.prepare_data_folder(TEST_SET1,TEST_SET2,data_path,X_train,X_val,fold_num) 

# print(train_dest_path)
# print(validation_dest_path)

# # # Model parameters
seed = 0
FACTOR = 2
BATCH_SIZE = 8 * FACTOR
EPOCHS = 1
HEIGHT = 224
WIDTH = 224
CHANNELS = 3


# Warm up paths

train_dest_path = '/home/faysal/PycharmProjects/DiabeticRetnopathy/TrainingPipeline/DR_Training_pipeline/data/warmup/train/'
validation_dest_path = '/home/faysal/PycharmProjects/DiabeticRetnopathy/TrainingPipeline/DR_Training_pipeline/data/warmup/validation/'


X_train = pd.read_csv('/home/faysal/PycharmProjects/DiabeticRetnopathy/TrainingPipeline/DR_Training_pipeline/data/warmup/train.csv')

X_val = pd.read_csv('/home/faysal/PycharmProjects/DiabeticRetnopathy/TrainingPipeline/DR_Training_pipeline/data/warmup/val.csv')

X_train = X_train.drop(['Unnamed: 0'],axis=1)
X_val = X_val.drop(['Unnamed: 0'],axis=1)

print(X_val['diagnosis'].unique())

X_train['diagnosis'] = X_train['diagnosis'].astype('str')
X_val['diagnosis'] = X_val['diagnosis'].astype('str')

# Call Generator
train_generator, valid_generator = generators.generator(X_train,train_dest_path,X_val,validation_dest_path,BATCH_SIZE,HEIGHT, WIDTH,seed)
print(train_generator, valid_generator)
# Call Model
models.model(X_train,EPOCHS,HEIGHT, WIDTH, CHANNELS,train_generator,valid_generator,weight_path,output_path)


# print(".............In main module.................")
# print()
# print(X_val)
# print(X_train)
# print(test)
# # print(class_weights)
# print(train_generator)
# print(valid_generator)


