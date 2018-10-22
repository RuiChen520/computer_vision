import numpy as np
import pydot
import keras.backend as K
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
K.set_image_data_format('channels_last')
from matplotlib.pyplot import imshow

class happymodel():
    def __init__(self, train=True, has_weights=False):
        # hyper parameter
        max_epoch = 40
        batch_size = 16

        X_train, y_train, X_test, y_test, classes = self.load_dataset()

        # Normalize image vectors
        self.X_train = X_train / 255.
        self.X_test = X_test / 255.

        # Reshape
        self.Y_train = y_train.T
        self.Y_test = y_test.T

        print("number of training examples = " + str(self.X_train.shape[0]))
        print("number of test examples = " + str(self.X_test.shape[0]))
        print("X_train shape: " + str(self.X_train.shape))
        print("Y_train shape: " + str(self.Y_train.shape))
        print("X_test shape: " + str(self.X_test.shape))
        print("Y_test shape: " + str(self.Y_test.shape))

        # create the model.
        self.model = self.BuildModel(X_train.shape[1:])
        # compile the model.
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        if train:
            self.train(max_epoch, batch_size, has_weights)

    def load_dataset(self):
        train_dataset = h5py.File('datasets/train_happy.h5', "r")
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])

        test_dataset = h5py.File('datasets/test_happy.h5', "r")
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])

        classes = np.array(test_dataset["list_classes"][:])

        train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
        test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

    def mean_pred(self, y_true, y_pred):
        return K.mean(y_pred)

    def BuildModel(self, input_shape):
        x_input = Input(input_shape)

        x = ZeroPadding2D((3, 3))(x_input)
        x = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(x)
        x = BatchNormalization(axis=3, name='btn0')(x)
        x = Activation('relu')(x)

        x = MaxPooling2D((2, 2), name='max_pool')(x)

        x = Flatten()(x)
        x = Dense(1, activation='sigmoid', name='fc')(x)

        model = Model(inputs=x_input, outputs=x)

        return model

    def train(self, max_epoch, batch_size, has_weights):
        if has_weights:
            self.model.load_weights('the_happy_house.h5')

        self.model.fit(x=self.X_train, y=self.Y_train, epochs=max_epoch, batch_size=batch_size)
        self.model.save_weights('the_happy_house.h5')

    def test(self):
        self.model.load_weights('the_happy_house.h5')
        preds = self.model.evaluate(x=self.X_test, y=self.Y_test)
        print()
        print("Loss = " + str(preds[0]))
        print("Test Accuracy = " + str(preds[1]))

    def predict(self, path):
        # 'images/test.jpg'
        img_path = path
        img = image.load_img(img_path, target_size=(64, 64))
        imshow(img)

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        self.model.load_weights('the_happy_house.h5')
        print(self.model.predict(x))


# ----------------------------------------------------------------
if __name__ == "__main__":
    model = happymodel(train=True, has_weights=False)
    model.test()
    model.predict('datasets/bad.jpg')
