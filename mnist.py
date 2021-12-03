import tensorflow as tf
from tensorflow.python import pywrap_tensorflow as _pywrap_tensorflow
from tensorflow.python.eager import context
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, Input, MaxPool2D, BatchNormalization, GlobalAvgPool2D

#print(tf.__version__)
#print(tf.config.list_physical_devices())


# building the classification model the sequential way:
model = tf.keras.Sequential(
    [
        Input(shape=(28,28,1)),
        Conv2D(32,(3,3),activation='relu'),
        Conv2D(64,(3,3),activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        Conv2D(128,(3,3),activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        GlobalAvgPool2D(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ]
)
def display_examples (examples, lables):
    plt.figure(figsize=(5,5))
    for i in range(20):
        indx = np.random.randint(0, examples.shape[0]-1)
        img = examples[indx]
        lable = lables[indx]

        plt.subplot(4,5 , i+1)
        plt.title(str(lable))
        plt.tight_layout()
        plt.imshow(img, cmap='gray')
    plt.show()

if __name__=='__main__':
    (x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print('x_train.shape = ' , x_train.shape)
    print('y_train.shape = ' , y_train.shape)
    print('x_test.shape = ' , x_test.shape)
    print('y_test.shape = ' , y_test.shape)

    if False:
        display_examples(x_train,y_train)

    # Normalizing the data before training: That's done through division
    # We divide by 255 because the colors channels are betwean 0 and 255
    # Before the division we need to convert it to float32 because it's represented as a 8bit int

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Now we expand the array dimensions from the 28 by 28 to the expected input layer shape of the model
    # axis = -1 is to add a dimension at the end of the array

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # Useful links for the hyperparameters of compiling a model:
    # Optimizers: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
    # Loss functions: https://www.tensorflow.org/api_docs/python/tf/keras/losses 
    # Metrics: https://www.tensorflow.org/api_docs/python/tf/keras/metrics 

    model.compile(optimizer= 'adam', loss= 'sparse_categorical_crossentropy', metrics= 'accuracy')
    model.fit(x_train,y_train, batch_size=64, epochs=3, validation_split=0.2)
    model.evaluate(x_test, y_test, batch_size=64)