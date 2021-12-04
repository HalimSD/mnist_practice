import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Input, MaxPool2D, BatchNormalization, GlobalAvgPool2D

# Building the classification model the functional way:
def functional_model():
    nn_input = Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3), activation='relu')(nn_input)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = GlobalAvgPool2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # To breduce a model that we can call the compile and fit function on it we
    # Have to tell tf that this is the model and pass it input and output as in:
    model = tf.keras.Model(inputs=nn_input, outputs=x)
    return model

# Building the classification model by inheriting from the tf.keras.Model() class:
class CustomModel(tf.keras.Model):
    # class instructor
    def __init__(self):
        super().__init__()
        # Creating the model layers:
        self.conv1 = Conv2D(32, (3, 3), activation='relu')
        self.conv2 = Conv2D(64, (3, 3), activation='relu')
        self.maxPool1 = MaxPool2D()
        self.batchNorm1 = BatchNormalization()

        self.conv3 = Conv2D(128, (3, 3), activation='relu')
        self.maxPool2 = MaxPool2D()
        self.batchNorm2 = BatchNormalization()

        self.globalAvgPool1 = GlobalAvgPool2D()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.maxPool1(x)
        x = self.batchNorm1(x)
        x = self.conv3(x)
        x = self.maxPool2(x)
        x = self.batchNorm2(x)
        x = self.globalAvgPool1(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x
