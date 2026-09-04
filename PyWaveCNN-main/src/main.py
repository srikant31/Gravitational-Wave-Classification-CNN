import os 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.image as img

import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping

from plotter import get_class_samples, model_accuracy_loss_plot, confusion_matrix_plot
from models import get_cnn_model, get_early_stopping
from data_handler import download_extract_split, check_files

def main():
    

    # Once run once you can set this to False
    if True:
        # Downloads, extracts and splits the data into train validation and test
        download_extract_split()

    # Visualise the types of data in the training data
    get_class_samples()
    
    # Get directories for image generator
    train_dir = '../data/train/'
    valid_dir = '../data/validation/'
    test_dir = '../data/test/'
    class_names = os.listdir(train_dir)

    img_size = (400, 400)
    batch_size = 32

    # Create the datasets for each directory
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        seed=220301,
        image_size=img_size,
        shuffle=True,
        batch_size=batch_size)

    valid_ds = tf.keras.utils.image_dataset_from_directory(
            valid_dir,
            seed=220301,
            image_size=img_size,
            batch_size=batch_size)

    test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            seed=220301,
            image_size=img_size,
            batch_size=batch_size)

    # Call CNN model
    cnn_model = get_cnn_model(img_size)

    # Print out the summary to view the model
    cnn_model.build((None, 400, 400, 3))
    cnn_model.summary()

    # Compile the model 
    cnn_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer='rmsprop', metrics=['accuracy'])

    # Get callback 
    early_stopping = get_early_stopping()

    # Fit the model to the training dataset and validate with the validation dataset.
    # Also apply the early stopping callback
    history = cnn_model.fit(train_ds,
                        epochs=20,
                        validation_data=valid_ds,
                        callbacks=[early_stopping])

    # Evaluate the model by giving our model some test data
    cnn_scores = cnn_model.evaluate(test_ds)

    # get the final accuracy from the test data
    final_accuracy = cnn_scores[1]
    model_accuracy_loss_plot(history)

    # Predict on the test dataset
    y_pred = []
    y_true = []
    for img_batch, label_batch in test_ds:
        preds = cnn_model.predict(img_batch)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(label_batch.numpy())

    # Plot a confusion matrix
    confusion_matrix_plot(y_true, y_pred, class_names)

    print(f'CNN Score: {round(final_accuracy, 4) * 100}%')  

if __name__ == '__main__':
    main()
