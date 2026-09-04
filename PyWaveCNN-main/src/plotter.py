import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.image as img
from sklearn.metrics import confusion_matrix
import seaborn as sns

def get_class_samples():

    # locate the training data
    train_dir = '../data/train/'
    class_names = os.listdir(train_dir)

    # Set up a grid with three columns
    n_cols = 3
    n_rows = int(np.ceil(len(class_names) / n_cols))

    # create the figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    plt.subplots_adjust(hspace=0.5, top=0.92, bottom=0.08)

    # Flatten the 2D array into a 1D array
    axes_flat = axes.flatten() if n_rows > 1 else [axes]

    # Configure where each plot is positioned
    for index, class_name in enumerate(class_names):
        ax = axes_flat[index]

        # Get the first image from each classes directory
        class_dir = os.path.join(train_dir, class_name)
        if os.path.isdir(class_dir):
            first_image_file = os.listdir(class_dir)[0]
            image_path = os.path.join(class_dir, first_image_file)

            # Plot the first image into its correct position
            if os.path.isfile(image_path):
                image = img.imread(image_path)
                ax.imshow(image)
                ax.set_title(class_name)
                ax.axis('off')

    # Remove any blank plots
    for i in range(len(class_names), len(axes_flat)):
        fig.delaxes(axes_flat[i])

    # Save figure
    fig.suptitle('Type of Gravitational Wave Detections', fontsize=24)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('../figures/gw_types.png', dpi=100)
    plt.close()

def model_accuracy_loss_plot(history):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='lower right')

    # Plot training & validation loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper right')

    # Save the plot
    plt.savefig('../figures/model_accuracy_loss_plot.png', dpi=100)
    plt.close()

def confusion_matrix_plot(y_true, y_pred, class_names):

    # Plot the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Get accuracy of the confustion matrix
    accuracy = np.trace(cm) / np.sum(cm) 
    title = (f"Confusion Matrix for Predictions - Accuracy: {accuracy:.2%}")

    # Configure Plot
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")  # Rotate labels and set horizontal alignment

    # Save confusion matrix
    plt.tight_layout() 
    plt.savefig('../figures/confusion_matrix_plot.png', dpi=100)
    plt.close()
