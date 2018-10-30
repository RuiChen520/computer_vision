from keras.applications.vgg19 import VGG19, preprocess_input
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.optimizers import SGD
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 训练和测试的图片分为'bus', 'dinosaur', 'flower', 'horse', 'elephant'五类
# 训练图片400张，测试图片100张；注意下载后，在train和test目录下分别建立上述的五类子目录，keras会按照子目录进行分类识别

NUM_CLASSES = 5
TRAIN_PATH = '/home/amax/PycharmProjects/reychan/re/train'
TEST_PATH = '/home/amax/PycharmProjects/reychan/re/test'

PREDICT_IMG = '/home/amax/PycharmProjects/reychan/re/test/dinosaur/402.jpg'

FC_NUMS = 1024

FREEZE_LAYERS = 17
IMAGE_SIZE = 224

base_model = VGG19(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')
print("VGG19 layer nums:", len(base_model.layers))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(FC_NUMS, activation='relu')(x)
prediction = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=prediction)
model.summary()
print("layer nums:", len(model.layers))

for layer in model.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in model.layers[FREEZE_LAYERS:]:
    layer.trainable = True
for layer in model.layers:
    print("layer.trainable:", layer.trainable)

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(directory=TRAIN_PATH,
                                                    target_size=(IMAGE_SIZE, IMAGE_SIZE), classes=['bus', 'dinosaur', 'flower', 'horse', 'elephant'])
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(directory=TEST_PATH,
                                                  target_size=(IMAGE_SIZE, IMAGE_SIZE), classes=['bus', 'dinosaur', 'flower', 'horse', 'elephant'])

model.fit_generator(train_generator, epochs=5, validation_data=test_generator)

img = load_img(path=PREDICT_IMG, target_size=(IMAGE_SIZE, IMAGE_SIZE))
x = img_to_array(img)
x = K.expand_dims(x, axis=0)
x = preprocess_input(x)

result = model.predict(x, steps=1)

class_x = K.eval(K.argmax(result))
classes = ['bus', 'dinosaur', 'flower', 'horse', 'elephant']
a = np.array(classes)[class_x]

test_image = mpimg.imread(PREDICT_IMG)
plt.imshow(test_image)
plt.show()
print("result:", a)