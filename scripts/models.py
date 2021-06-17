import os
import sys
import cv2
import shutil
import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import multiprocessing as mp
import matplotlib.pyplot as plt
# from tensorflow import set_random_seed
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from keras import backend as K
from keras.models import Model
from keras.models import load_model
from keras.utils import to_categorical
from keras import optimizers, applications
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, LearningRateScheduler

# def seed_everything(seed=0):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     set_random_seed(0)

seed = 0
# seed_everything(seed)
# %matplotlib inline
# sns.set(style="whitegrid")
# warnings.filterwarnings("ignore")
from keras_efficientnets import EfficientNetB5

# Parameters

def model(X_train,EPOCHS,HEIGHT, WIDTH, CHANNELS,train_generator,valid_generator,weight_path,output_path):
    # seed = 0
    FACTOR = 2
    BATCH_SIZE = 8 * FACTOR
    EPOCHS = 10
    WARMUP_EPOCHS = 2
    LEARNING_RATE = 1e-4 * FACTOR
    WARMUP_LEARNING_RATE = 1e-3 * FACTOR
    # HEIGHT = 224
    # WIDTH = 224
    # CHANNELS = 3
    TTA_STEPS = 3
    ES_PATIENCE = 4
    RLROP_PATIENCE = 3
    DECAY_DROP = 0.5
    LR_WARMUP_EPOCHS_1st = 2
    LR_WARMUP_EPOCHS_2nd = 4
    STEP_SIZE = len(X_train) // BATCH_SIZE
    TOTAL_STEPS_1st = WARMUP_EPOCHS * STEP_SIZE
    TOTAL_STEPS_2nd = EPOCHS * STEP_SIZE
    WARMUP_STEPS_1st = LR_WARMUP_EPOCHS_1st * STEP_SIZE
    WARMUP_STEPS_2nd = LR_WARMUP_EPOCHS_2nd * STEP_SIZE

    def cosine_decay_with_warmup(global_step,
                                 learning_rate_base,
                                 total_steps,
                                 warmup_learning_rate=0.0,
                                 warmup_steps=0,
                                 hold_base_rate_steps=0):
        if total_steps < warmup_steps:
            raise ValueError('total_steps must be larger or equal to warmup_steps.')
        learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
            np.pi *
            (global_step - warmup_steps - hold_base_rate_steps
             ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
        if hold_base_rate_steps > 0:
            learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                     learning_rate, learning_rate_base)
        if warmup_steps > 0:
            if learning_rate_base < warmup_learning_rate:
                raise ValueError('learning_rate_base must be larger or equal to warmup_learning_rate.')
            slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
            warmup_rate = slope * global_step + warmup_learning_rate
            learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                     learning_rate)
        return np.where(global_step > total_steps, 0.0, learning_rate)


    class WarmUpCosineDecayScheduler(Callback):
        """Cosine decay with warmup learning rate scheduler"""

        def __init__(self,
                     learning_rate_base,
                     total_steps,
                     global_step_init=0,
                     warmup_learning_rate=0.0,
                     warmup_steps=0,
                     hold_base_rate_steps=0,
                     verbose=0):
            

            super(WarmUpCosineDecayScheduler, self).__init__()
            self.learning_rate_base = learning_rate_base
            self.total_steps = total_steps
            self.global_step = global_step_init
            self.warmup_learning_rate = warmup_learning_rate
            self.warmup_steps = warmup_steps
            self.hold_base_rate_steps = hold_base_rate_steps
            self.verbose = verbose
            self.learning_rates = []

        def on_batch_end(self, batch, logs=None):
            self.global_step = self.global_step + 1
            lr = K.get_value(self.model.optimizer.lr)
            self.learning_rates.append(lr)

        def on_batch_begin(self, batch, logs=None):
            lr = cosine_decay_with_warmup(global_step=self.global_step,
                                          learning_rate_base=self.learning_rate_base,
                                          total_steps=self.total_steps,
                                          warmup_learning_rate=self.warmup_learning_rate,
                                          warmup_steps=self.warmup_steps,
                                          hold_base_rate_steps=self.hold_base_rate_steps)
            K.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print('\nBatch %02d: setting learning rate to %s.' % (self.global_step + 1, lr))

    def create_model(input_shape):
        input_tensor = Input(shape=input_shape)
        base_model = EfficientNetB5(weights=None, 
                                    include_top=False,
                                    input_tensor=input_tensor)
        base_model.load_weights(weight_path)

        x = GlobalAveragePooling2D()(base_model.output)
        final_output = Dense(5, activation='softmax', name='final_output')(x)
        model = Model(input_tensor, final_output)
    
        return model


    # Train top layers
    model = create_model(input_shape=(HEIGHT, WIDTH, CHANNELS))

    for layer in model.layers:
        layer.trainable = False

    for i in range(-2, 0):
        model.layers[i].trainable = True

    cosine_lr_1st = WarmUpCosineDecayScheduler(learning_rate_base=WARMUP_LEARNING_RATE,
                                               total_steps=TOTAL_STEPS_1st,
                                               warmup_learning_rate=0.0,
                                               warmup_steps=WARMUP_STEPS_1st,
                                               hold_base_rate_steps=(2 * STEP_SIZE))

    metric_list = ["accuracy"]
    callback_list = [cosine_lr_1st]
    optimizer = optimizers.Adam(lr=WARMUP_LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=metric_list)

    STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size


    history_warmup = model.fit_generator(generator=train_generator,
                                         steps_per_epoch=STEP_SIZE_TRAIN,
                                         validation_data=valid_generator,
                                         validation_steps=STEP_SIZE_VALID,
                                         epochs=2,
                                         callbacks=callback_list,
                                         # class_weight=class_weights,
                                         verbose=1).history

    # Fine-tune the complete model

    for layer in model.layers:
        layer.trainable = True

    es = EarlyStopping(monitor='val_loss', mode='min', patience=ES_PATIENCE, restore_best_weights=True, verbose=1)
    cosine_lr_2nd = WarmUpCosineDecayScheduler(learning_rate_base=LEARNING_RATE,
                                               total_steps=TOTAL_STEPS_2nd,
                                               warmup_learning_rate=0.0,
                                               warmup_steps=WARMUP_STEPS_2nd,
                                               hold_base_rate_steps=(3 * STEP_SIZE))

    callback_list = [es, cosine_lr_2nd]
    optimizer = optimizers.Adam(lr=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=metric_list)

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=STEP_SIZE_TRAIN,
                                  validation_data=valid_generator,
                                  validation_steps=STEP_SIZE_VALID,
                                  epochs=10,
                                  # class_weight=class_weights,
                                  callbacks=callback_list,
                                  verbose=1).history

    # Model loss curve
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(20, 14))

    ax1.plot(history['loss'], label='Train loss')
    ax1.plot(history['val_loss'], label='Validation loss')
    ax1.legend(loc='best')
    ax1.set_title('Loss')

    ax2.plot(history['accuracy'], label='Train accuracy')
    ax2.plot(history['val_accuracy'], label='Validation accuracy')
    ax2.legend(loc='best')
    ax2.set_title('Accuracy')

    plt.xlabel('Epochs')
    sns.despine()
    fig.savefig(output_path+'/Model loss curve.png')
    plt.close(fig)

    # Model Evaluation

    N_CLASSES = 5
    lastFullTrainPred = np.empty((0, N_CLASSES))
    lastFullTrainLabels = np.empty((0, N_CLASSES))
    lastFullValPred = np.empty((0, N_CLASSES))
    lastFullValLabels = np.empty((0, N_CLASSES))

    for i in range(STEP_SIZE_TRAIN+1):
        im, lbl = next(train_generator)
        scores = model.predict(im, batch_size=train_generator.batch_size)
        lastFullTrainPred = np.append(lastFullTrainPred, scores, axis=0)
        lastFullTrainLabels = np.append(lastFullTrainLabels, lbl, axis=0)

    for i in range(STEP_SIZE_VALID+1):
        im, lbl = next(valid_generator)
        scores = model.predict(im, batch_size=valid_generator.batch_size)
        lastFullValPred = np.append(lastFullValPred, scores, axis=0)
        lastFullValLabels = np.append(lastFullValLabels, lbl, axis=0)

    train_preds = [np.argmax(pred) for pred in lastFullTrainPred]
    train_labels = [np.argmax(label) for label in lastFullTrainLabels]
    validation_preds = [np.argmax(pred) for pred in lastFullValPred]
    validation_labels = [np.argmax(label) for label in lastFullValLabels]

    # Confusion Matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex='col', figsize=(24, 7))
    labels = ['0 - No DR', '1 - Mild', '2 - Moderate', '3 - Severe', '4 - Proliferative DR']
    train_cnf_matrix = confusion_matrix(train_labels, train_preds)
    validation_cnf_matrix = confusion_matrix(validation_labels, validation_preds)

    train_cnf_matrix_norm = train_cnf_matrix.astype('float') / train_cnf_matrix.sum(axis=1)[:, np.newaxis]
    validation_cnf_matrix_norm = validation_cnf_matrix.astype('float') / validation_cnf_matrix.sum(axis=1)[:, np.newaxis]

    train_df_cm = pd.DataFrame(train_cnf_matrix_norm, index=labels, columns=labels)
    validation_df_cm = pd.DataFrame(validation_cnf_matrix_norm, index=labels, columns=labels)

    sns.heatmap(train_df_cm, annot=True, fmt='.2f', cmap="Blues",ax=ax1).set_title('Train')
    sns.heatmap(validation_df_cm, annot=True, fmt='.2f', cmap="Blues",ax=ax2).set_title('Validation')
    
    fig.savefig(output_path+'/Confusion Matrix.png')
    plt.close(fig)

    print("Train Cohen Kappa score: %.3f" % cohen_kappa_score(train_preds, train_labels, weights='quadratic'))
    print("Validation Cohen Kappa score: %.3f" % cohen_kappa_score(validation_preds, validation_labels, weights='quadratic'))
    print("Complete set Cohen Kappa score: %.3f" % cohen_kappa_score(train_preds+validation_preds, train_labels+validation_labels, weights='quadratic'))

        
    # Save weights
    model.save_weights(output_path+'effNetB5_weights_newpreprocessedimg224_fold4.h5')

    # Save model
    model.save(output_path+'b5_newpreprocessed_full_fold4.h5')

