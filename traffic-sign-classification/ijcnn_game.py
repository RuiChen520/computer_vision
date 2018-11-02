import numpy as np
from skimage import io, color, exposure, transform
from sklearn.cross_validation import train_test_split
import os
import glob
import h5py
import matplotlib.image as mpimg
import pandas as pd
from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import backend as K
K.set_image_data_format('channels_last')

from matplotlib import pyplot as plt


NUM_CLASSES = 43
IMG_SIZE = 48

def preprocess_img(img):
    # Histogram normalization in y
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    # central scrop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
    img = np.rollaxis(img, -1)

    return img

def get_class(img_path):
    return int(img_path.split('/')[-2])


try:
    with  h5py.File('X.h5') as hf:
        X, Y = hf['imgs'][:], hf['labels'][:]
        X = X.transpose((0, 3, 2, 1))
    print(X.shape,Y.shape)
    print("Loaded images from X.h5")

except (IOError, OSError, KeyError):
    print("Error in reading X.h5. Processing all images...")
    root_dir = '/home/amax/PycharmProjects/reychan/GTSRB-Training_fixed/GTSRB/Training'
    imgs = []
    labels = []

    all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
    np.random.shuffle(all_img_paths)
    for img_path in all_img_paths:
        try:
            img = preprocess_img(io.imread(img_path))
            label = get_class(img_path)
            imgs.append(img)
            labels.append(label)

            if len(imgs) % 1000 == 0: print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
        except (IOError, OSError):
            print('missed', img_path)
            pass

    X = np.array(imgs, dtype='float32')
    Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]

    with h5py.File('X.h5', 'w') as hf:
        hf.create_dataset('imgs', data=X)
        hf.create_dataset('labels', data=Y)

def cnn_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(IMG_SIZE, IMG_SIZE, 3),
                     activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model

model = cnn_model()
# let's train the model using SGD + momentum (how original).
model.summary()
lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
          optimizer=sgd,
          metrics=['accuracy'])


def lr_schedule(epoch):
    return lr*(0.1**int(epoch/10))

batch_size = 32
nb_epoch = 30

labels_dense=np.asarray(Y)
print(labels_dense.shape)
num_labels = labels_dense.shape[0]
index_offset = np.arange(num_labels) * NUM_CLASSES
labels_one_hot = np.zeros((num_labels, NUM_CLASSES))
labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
Y = labels_one_hot
print(Y.shape)

#model.fit(X, Y, batch_size=batch_size, epochs=nb_epoch, validation_split=0.2, shuffle=True, callbacks=[LearningRateScheduler(lr_schedule), ModelCheckpoint('model.h5',save_best_only=True)] )
#model.save_weights('model.h5')
model.load_weights('model.h5')


#test = pd.read_csv('/home/amax/PycharmProjects/reychan/GTSRB/Final_Test/Images/GT-final_test.csv', sep=';')
with  h5py.File('X_test.h5') as hf5:
    X_test, y_test = hf5['imgs'][:], hf5['labels'][:]
    X_test = X_test.transpose((0, 3, 2, 1))
print(X_test.shape, y_test.shape)
print("Loaded images from X_test.h5")

X_test = np.array(X_test)
y_test = np.array(y_test)
y_pred = model.predict_classes(X_test)
acc = np.sum(y_pred==y_test)/np.size(y_pred)
print(np.size(y_pred),np.size(y_test))
print(np.sum(y_pred==y_test))
print("Test accuracy = {}".format(acc))

