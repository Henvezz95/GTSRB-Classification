import cv2, os, time, random
import numpy as np 
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LeakyReLU, Dense, Dropout, Input, BatchNormalization, concatenate, Activation 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

from matplotlib.image import imread
import pyfastnoisesimd as fns

def add_noise(img):
    img_shape = img.shape
    std = np.random.uniform(0.0, 0.06)
    return img + np.random.normal(0, std, img_shape)

def elastic_transform(img, strength, frequency, octaves, lacunarity):
    img_shape = img.shape
    perlin = fns.Noise(seed=np.random.randint(0, 2**31))
    perlin.frequency = frequency
    perlin.noiseType = fns.NoiseType.PerlinFractal
    perlin.fractal.octaves = octaves
    perlin.fractal.lacunarity = lacunarity
    sdx, sdy = perlin.genAsGrid((img_shape[0], img_shape[1]))*strength, perlin.genAsGrid((img_shape[0], img_shape[1]))*strength
    x_grid, y_grid = np.meshgrid(np.arange(img_shape[1]), np.arange(img_shape[0]))
    remap_y, remap_x = y_grid.astype('float32') + sdy, x_grid.astype('float32') + sdx
    for i in range(img_shape[2]):
        img[:,:,i] = cv2.remap(img[:,:,i], remap_x, remap_y, cv2.INTER_CUBIC)
    return img

def function(img):
    result = add_noise(img)
    strength = np.random.uniform(1, 4)
    frequency = np.random.uniform(0.02, 0.09)
    octaves = 3
    lacunarity = 2.0
    result = elastic_transform(result, strength=strength, frequency=frequency, octaves=octaves, lacunarity=lacunarity)
    return result

def cnn_model(l2_conv=0.0, l2_dense=0.0):
   
    main_input = Input(shape=(32, 32, 3), dtype='float32', name='main_input')
    
    x = Conv2D(32, (5, 5), padding='same', input_shape=(32, 32, 3), activation='relu', kernel_regularizer=l2(l2_conv)) (main_input)
    x = Conv2D(32, (5, 5), padding='same', activation='relu', kernel_regularizer=l2(l2_conv)) (x)
    x = MaxPooling2D(pool_size=(2, 2)) (x)
    
    y = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(l2_conv)) (x)
    y = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(l2_conv)) (y)
    y = MaxPooling2D(pool_size=(2, 2)) (y)
    
    z = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(l2_conv)) (y)
    z = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(l2_conv)) (z)
    z = MaxPooling2D(pool_size=(2, 2)) (z)
    
    x1 = MaxPooling2D(pool_size=(4, 4)) (x)
    x1 = Flatten()(x1)
    y1 = MaxPooling2D(pool_size=(2, 2)) (y)
    y1 = Flatten()(y1)
    z1 = Flatten()(z)
    
    CNN_output = concatenate([x1,y1,z1])
    CNN_output = Dropout(0.5)(CNN_output)
    CNN_output = Dense(512, activation='relu', kernel_regularizer=l2(l2_dense))(CNN_output)
    CNN_output = Dropout(0.25)(CNN_output)
    CNN_output = Dense(43, activation='softmax', kernel_regularizer=l2(l2_dense))(CNN_output)
    
    model = Model(inputs=[main_input], outputs=[CNN_output])
    return model

# Check if GPU is detected
device_name = tf.test.gpu_device_name()
if "GPU" not in device_name:
    print("GPU device not found")
print('Found GPU at: {}'.format(device_name))

seed = 31
train_paths = glob('Train/*/*.png')
num_train_images = len(train_paths)
num_classes = len(os.listdir('Train/'))

X = np.zeros((num_train_images, 32, 32, 3), dtype=np.float32)
y = np.zeros((num_train_images, num_classes), dtype=np.float32)

for i in range(num_train_images):
    img = cv2.imread(train_paths[i])
    img = cv2.resize(img, (32,32), cv2.INTER_CUBIC)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=10.0,tileGridSize=(4,4))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    X[i] = bgr/255.0

    class_id = int(os.path.basename(os.path.dirname(train_paths[i])))
    y[i][class_id] = 1
    print('\r'+'['+str(i+1)+'/'+str(num_train_images)+']',end="")

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.25, random_state=seed)

generator_train = tf.keras.preprocessing.image.ImageDataGenerator(shear_range=0.2,
                                                            zoom_range=0.15,
                                                            rotation_range=10,
                                                            width_shift_range=4,
                                                            height_shift_range=4,
#                                                            fill_mode='reflect',
                                                            preprocessing_function=function
)
generator_val = tf.keras.preprocessing.image.ImageDataGenerator()

train_gen = generator_train.flow(X_train, y_train)
val_gen = generator_val.flow(X_val, y_val)

batchSize = 128
num_train = len(X_train)
num_val = len(X_val)
classifier = cnn_model(l2_conv=0.0002)
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(train_gen,
                steps_per_epoch = (num_train/batchSize),
                epochs = 300,
                validation_data = val_gen,
                validation_steps = (num_val/batchSize))

#Fine Tuning
sgd = tf.keras.optimizers.SGD(
    learning_rate=0.00005, momentum=0.9, nesterov=False
)


save_path = './classifier.h5'
check_pointer = ModelCheckpoint(save_path, save_best_only=True, monitor="val_loss")

batchSize = 512
classifier.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(train_gen,
                steps_per_epoch = (num_train/batchSize),
                epochs = 300,
                validation_data = val_gen,
                validation_steps = (num_val/batchSize),
                callbacks=[check_pointer])