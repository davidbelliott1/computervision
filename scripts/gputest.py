import tensorflow as tf

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# reshape and rescale data for the CNN

train_images = train_images.reshape(60000, 28, 28, 1)

test_images = test_images.reshape(10000, 28, 28, 1)

train_images, test_images = train_images/255, test_images/255

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

tensor_board = tf.keras.callbacks.TensorBoard('./logs/LeNet-MNIST-1')

import time

start_time=time.time()

model.fit(train_images, train_labels, batch_size=128, epochs=15, verbose=1,
         validation_data=(test_images, test_labels), callbacks=[tensor_board])

print('Training took {} seconds'.format(time.time()-start_time))
