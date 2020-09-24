#!/usr/bin/env python3

import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import model_from_json
import numpy as np
import os
import sys
from skimage.io import imread, imshow

# load json and create model
json_file = open('model.json', 'r')
model = json_file.read()
json_file.close()
model = model_from_json(model)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
#model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#score = model.evaluate(X, Y, verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

imgpath = sys.argv[1]
print("reading image", imgpath)
img = imread(imgpath)
img2 = tf.image.resize(img, (128, 128))
x = np.array(img2)
x = np.expand_dims(x, axis=0)
predict = model.predict(x, verbose=1)

predict = (predict > 0.7).astype(np.uint8)

imshow(np.squeeze(predict[0]))
plt.show()

imshow(img)
plt.show()

