#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys
from skimage.io import imread, imshow
from tensorflow.keras.models import load_model


# load model
model = load_model('model_with_weights.h5')
print("Loaded model from disk")

imgpath = sys.argv[1]
print("reading image", imgpath)
img = imread(imgpath)
img2 = tf.image.resize(img, (224, 224))
x = np.array(img2)
x = np.expand_dims(x, axis=0)
predict = model.predict(x, verbose=1)
predict = (predict > 0.5).astype(np.uint8)
predict = np.squeeze(predict[0])
print(predict.shape)
print(predict)

image = Image.open(imgpath)
mask = Image.fromarray(predict * 255, "L")
mask = mask.resize(image.size)

#translucent mask
mask = mask.convert("RGBA")
datas = mask.getdata()
white = (255, 255, 255, 0)
newData = []
for item in datas:
    if item[0] == 255 and item[1] == 255 and item[2] == 255:
        newData.append((255, 255, 255, 0))
    else:
        newData.append(item)
mask.putdata(newData)
mask.show()

new_img = image.copy()
new_img = new_img.convert("RGBA")
new_img.paste(mask, (0, 0), mask)
#new_img = new_img.convert("RGB")
new_img.show()
