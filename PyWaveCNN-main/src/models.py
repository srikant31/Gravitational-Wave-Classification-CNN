import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping

def get_cnn_model(img_size):
    
    # Resize and rescale the images coming into the model
    resize_and_rescale = tf.keras.Sequential([
        layers.Resizing(img_size[0], img_size[1]),
        layers.Rescaling(1./255)
        ])
    
    # Convolutional Neural Network
    model = tf.keras.models.Sequential([
        
        # Apply our resize and rescale
        resize_and_rescale,

        # Convolutional layers and Pooling layers
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(img_size + (3,))),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Flatten into a 1D array to feed into a Deep Neural Network
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(22, activation='softmax')
        ])

    return model

def get_early_stopping():
    
    # Used to stop the CNN early if baseline is met
    early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            mode='max',
            restore_best_weights=True,
            verbose=1,
            baseline=0.95)

    return early_stopping

