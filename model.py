#!/usr/bin/env python3

# https://androidkt.com/tensorflow-keras-unet-for-image-image-segmentation/

import tensorflow as tf
import os
import sys
import glob
import numpy as np
import random
from tqdm import tqdm
from itertools import chain
from skimage.transform import resize
from skimage.io import imread, imshow

# tf.enable_eager_execution()

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

DATA_PATH = "images"

seed = 42
random.seed = seed
np.random.seed = seed

images = glob.glob(DATA_PATH + "/*.jpg")
masks = glob.glob(DATA_PATH + "/*_mask.png")
print(len(images), len(masks))

X = np.zeros((len(images), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y = np.zeros((len(images), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

for n, (imagefn, maskfn) in tqdm(enumerate(zip(images, masks)), total=len(images)):
    img = imread(imagefn)[:, :, :IMG_CHANNELS]
    img = tf.image.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    X[n] = img
    mask = imread(maskfn)
    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH))
    mask = np.expand_dims(mask, axis=-1)
    Y[n] = mask

x_train = X
y_train = Y


# Build U-Net model
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

args = {
    "activation": tf.keras.activations.elu,
    "kernel_initializer": "he_normal",
    "padding": "same",
}

c1 = tf.keras.layers.Conv2D(16, (3, 3), **args)(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), **args)(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), **args)(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), **args)(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3, 3), **args)(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), **args)(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

c4 = tf.keras.layers.Conv2D(128, (3, 3), **args)(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), **args)(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

c5 = tf.keras.layers.Conv2D(256, (3, 3), **args)(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), **args)(c5)

u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), **args)(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), **args)(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), **args)(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), **args)(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), **args)(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), **args)(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), **args)(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), **args)(c9)

outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(c9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])


checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, save_weights_only=True, verbose=1
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor="val_loss"),
    tf.keras.callbacks.TensorBoard(log_dir="./logs"),
    cp_callback,
]

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

results = model.fit(
    x_train,
    y_train,
    validation_split=0.1,
    batch_size=64,
    epochs=20,
    callbacks=callbacks,
)

model.summary()

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
