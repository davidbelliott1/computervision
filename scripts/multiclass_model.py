# script version of the Multiclass Modeling notebook

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils, to_categorical
from keras.backend.tensorflow_backend import set_session

# height and width of images, length of flattened array
width = height = 28
length = width * height

mapping_dict = {
        'articulated_truck' : 0,
        'background' : 1,
        'bicycle' : 2,
        'bus' : 3,
        'car' : 4,
        'motorcycle' : 5,
        'non-motorized_vehicle' : 6,
        'pedestrian' : 7,
        'pickup_truck' : 8,
        'single_unit_truck' : 9,
        'work_van' : 10
    }

# save file name. If you needed a comment to figure this one out, go lie down and put a wet towel on your head
img_save_file = ('./data/imagefile.npy')
target_save_file = ('./data/targetfile.npy')

# Load the data files
X = np.load(img_save_file)
image_target = np.load(target_save_file)

print(X.shape)

# Load the targets into a dataframe, map for the categories, create y
target_df = pd.DataFrame(image_target)
y = target_df[0].map(mapping_dict)
print(len(y))

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

# Normalize
X_train = X_train.astype('float') / 255
X_test = X_test.astype('float') / 255

# Reshape features back into stacked matrices
X_train_new = X_train.reshape(X_train.shape[0],width, height, 1)
X_test_new = X_test.reshape(X_test.shape[0], width, height, 1)

# change the y into categorical
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

print(X_train_new.shape)
print(X_test_new.shape)
print(y_train_cat.shape)
print(y_test_cat.shape)


# This bit of code allows keras/tf to dynamically grow the GPU memory. This may or may not be solving the GPU
# issue I was having. Taken from https://www.cicoria.com/keras-tensor-flow-cublas_status_not_initialized/


config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

# Instantiate
cnn_model = Sequential()

#Input Layer
cnn_model.add(Conv2D(filters=112, kernel_size=(3,3), activation='relu',input_shape = (width, height,1)))
cnn_model.add(MaxPooling2D(pool_size=2))
cnn_model.add(Dropout(0.3))

# Second Layer
cnn_model.add(Conv2D(56, kernel_size=(3,3),activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=2))
cnn_model.add(Dropout(0.3))


# Third Layer
cnn_model.add(Conv2D(28, kernel_size=(2,2),activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=2))
cnn_model.add(Dropout(0.3))

# Fourth Layer
cnn_model.add(Conv2D(14, kernel_size=2,activation='relu'))
# cnn_model.add(MaxPooling2D(pool_size=2))
cnn_model.add(Dropout(0.5))

# Flatten
cnn_model.add(Flatten())

# Fifth Layer
cnn_model.add(Dense(112, activation='relu'))

# Sixth Layer
cnn_model.add(Dense(56, activation='relu'))

# Seventh Layer
cnn_model.add(Dense(28, activation='relu'))

# Output Layer
cnn_model.add(Dense(11, activation='softmax'))


# Compile
cnn_model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

# Fit
history = cnn_model.fit(X_train_new,
                        y_train_cat,
                        batch_size=256,
                        validation_data=(X_test_new, y_test_cat),
                        epochs=1,
                        verbose=True)

# Check out our train loss and test loss over epochs.
train_loss = history.history['loss']
test_loss = history.history['val_loss']

# Set figure size.
plt.figure(figsize=(12, 8))

# Generate line plot of training, testing loss over epochs.
plt.plot(train_loss, label='Training Loss', color='#185fad')
plt.plot(test_loss, label='Testing Loss', color='orange')

# Set title
plt.title('Training and Testing Loss by Epoch', fontsize = 25)
plt.xlabel('Epoch', fontsize = 18)
plt.ylabel('Categorical Crossentropy', fontsize = 18)
plt.xticks(range(10))

plt.legend(fontsize = 18);
plt.savefig('./1982x28x1010epohcs.png')

columns_dict = {
        0 :'articulated_truck',
        1: 'background',
        2:'bicycle',
        3:'bus',
        4:'car',
        5:'motorcycle',
        6:'non-motorized_vehicle',
        7:'pedestrian',
        8:'pickup_truck',
        9:'single_unit_truck',
        10:'work_van'
    }
cf = pd.DataFrame(metrics.confusion_matrix(y_test, predictions))

cf.rename(columns=columns_dict, inplace=True)
cf.rename(index=columns_dict, inplace=True)
cf