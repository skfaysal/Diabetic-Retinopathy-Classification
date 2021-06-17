import pandas as pd
from sklearn.utils import class_weight
import os
import numpy as np
from sklearn.utils import class_weight

"""
# check what contains in the aptos, eyepacs and usydeny dataset

# load  5 fold data and check levels

# Load X_train, X_val

# Conevrt diagnosis column in string

# Change the image format from .jpeg into jpg

# Define Test set from Data_3 or Usydeny

# Claculate Class weights
"""

"""
INPUT: Data_1,Data_2,Data_3,fold_data,fold_num


OUTPUT: X_train,X_val,test,class_weights

"""

def prepare_df(Data_1,Data_2,Data_3,fold_data,fold_num):
	# Check all the csv's aptos, eyepacs, usydney. and take the last usydney
	for filename in [Data_1, Data_2, Data_3]:
		data = pd.read_csv(filename)
		sp=filename.split('/')
		print("[INFO]shape of dataframe "+str(sp[-2])+":",data.shape)
		print("[INFO]columns of dataframe "+str(sp[-2])+":",data.columns)
		print()

	# Read fold_data csv
	fold_set = pd.read_csv(fold_data)

	# Print the values of levels of each column
	for i in fold_set.columns:
		print("[INFO] count for: ",i)
		print(fold_set[i].value_counts())
		print()

	# Get x_train and x_val from fold4
	X_train = fold_set[fold_set[fold_num] == 'train']
	X_val = fold_set[fold_set[fold_num] == 'validation']
	    

	X_train['diagnosis'] = X_train['diagnosis'].astype('str')
	X_val['diagnosis'] = X_val['diagnosis'].astype('str')


	# Convert .jpeg into .jpg
	# Since 5 fold csv contain .jpeg format 
	for data in X_train['id_code']:
	    tmp= data
	    data=data.replace('.jpeg', '.jpg')
	    X_train.id_code[X_train.id_code == tmp] = data

	for data in X_val['id_code']:
	    tmp= data
	    data=data.replace('.jpeg', '.jpg')
	    X_val.id_code[X_val.id_code == tmp] = data


	# Claculate class weights for imbalanced dataset
	class_weights = class_weight.compute_class_weight('balanced',
	                                                 np.unique(X_train["diagnosis"]),
	                                                 X_train["diagnosis"])
	print("[INFO] Class weights: ",class_weights)

	# Get test set from Data_3 or Usydney data
	test = pd.read_csv(Data_3)

	# Convert all test set images/Usydney data into .jpeg format. Since no format is defined in the csv
	test["id_code"] = test["id_code"].apply(lambda x: x + ".jpeg")

	# Check dimension of our data
	print('[INFO] Number of train samples: ', X_train.shape[0])
	print('[INFO] Number of validation samples: ', X_val.shape[0])
	print('[INFO] Number of test samples: ', test.shape[0])

	return X_train,X_val,test,class_weights