import pandas as pd
import os
import numpy as np
import shutil

def prepare_data_folder(TEST_SET1,TEST_SET2,data_path,X_train,X_val,fold_num):

    dataset_one = TEST_SET1
    dataset_two = TEST_SET2


    train_dest_path = (str(data_path)+'/train_images/')
    validation_dest_path = (str(data_path)+'/validation_images/')

    os.makedirs(validation_dest_path)
    os.makedirs(train_dest_path)

    def copy_files_to_temp_folder(df):
        df = df.reset_index()
        for i in range(df.shape[0]):
                item = df.iloc[i]
                image_id = item['id_code']
                item_set = item[fold_num]
                item_data = item['data']
                if item_set == 'train':
                    if item_data == 'new':
                        copy_image(image_id, dataset_one, train_dest_path)
                    if item_data == 'old':
                        copy_image(image_id, dataset_two, train_dest_path)
                if item_set == 'validation':
                    if item_data == 'new':
                        copy_image(image_id, dataset_one, validation_dest_path)
                    if item_data == 'old':
                        copy_image(image_id, dataset_two, validation_dest_path)

    def copy_image(file, src, dest):
        src_file = src + str(file)
        dest_file = dest + str(file)
        # print(src_file)
        # print(dest_file)
        # print(os.path.isfile(src_file))
        shutil.copyfile(src_file, dest_file)

    # Call function
    copy_files_to_temp_folder(X_val)
    copy_files_to_temp_folder(X_train)

    print("[INFO] Train and Validation folder successfully created")
    


    return train_dest_path,validation_dest_path