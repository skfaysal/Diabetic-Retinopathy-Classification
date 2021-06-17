
"""
INPUT: X_train,train_dest_path,X_val,validation_dest_path,test,test_path,BATCH_SIZE,HEIGHT, WIDTH,seed,

OUTPUT: train_generator, valid_generator,test_generator
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def generator(X_train,train_dest_path,X_val,validation_dest_path,BATCH_SIZE,HEIGHT, WIDTH,seed):

    datagen=ImageDataGenerator(rescale=1./255, 
                               rotation_range=360,
                               horizontal_flip=True,
                               vertical_flip=True)

    train_generator=datagen.flow_from_dataframe(
                            dataframe=X_train,
                            directory=train_dest_path,
                            x_col="id_code",
                            y_col="diagnosis",
                            class_mode="categorical",
                            batch_size=BATCH_SIZE,
                            target_size=(HEIGHT, WIDTH),
                            seed=seed)

    valid_generator=datagen.flow_from_dataframe(
                            dataframe=X_val,
                            directory=validation_dest_path,
                            x_col="id_code",
                            y_col="diagnosis",
                            class_mode="categorical",
                            batch_size=BATCH_SIZE,
                            target_size=(HEIGHT, WIDTH),
                            seed=seed)

    # test_generator=datagen.flow_from_dataframe(  
    #                        dataframe=test,
    #                        directory=test_path,
    #                        x_col="id_code",
    #                        batch_size=1,
    #                        class_mode=None,
    #                        shuffle=False,
    #                        target_size=(HEIGHT, WIDTH),
    #                        seed=seed)

    return train_generator, valid_generator