from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import imageprocessing.ImageProcessing as IP
import time
import os

start_time = time.time()

dim = (224, 224)
base_model = keras.applications.VGG16(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(224, 224, 3),
    include_top=False)  # Do not include the ImageNet classifier at the top.
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs)
x = keras.layers.Flatten()(x)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)
model.load_weights('D:\\MA_Luisa\\models\\FINAL_2\\all_20210612_155018\\vgg_ckpt3\\vgg_MaskedSquaredHalf_2_3')
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.TruePositives(),
                       tf.keras.metrics.TrueNegatives(),
                       tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()])

print("--- Load Model %s seconds ---\n" % (time.time() - start_time))
start_time = time.time()

croppedImages = []
startProc = time.time()
for imgpath in os.listdir('D:\\MA_Luisa\\data\\exportKRLM_2_Half\\abnormal\\2007131650IA_A\\2'):
    img = cv2.imread('D:\\MA_Luisa\\data\\exportKRLM_2_Half\\abnormal\\2007131650IA_A\\2\\' + imgpath, cv2.IMREAD_UNCHANGED)

    img = IP.OutlierRemovalBasedOnMean(img)
    mask = IP.DoAll(img)
    res = IP.ApplyMask(img, mask)
    res = IP.DeleteRows(res)

    height, width = res.shape
    min_shape = min(width, height)
    max_shape = max(width, height)
    imgCount = int(np.floor(max_shape/min_shape))
    overlap = (min_shape - (max_shape - min_shape * imgCount))/imgCount

    for i in range(imgCount + 1):
        startingPoint = int(np.floor((i*min_shape) - (i*overlap)))
        if max_shape == width:
            imgCropped = res[0:min_shape, startingPoint:(startingPoint + min_shape)]
        else:
            imgCropped = res[startingPoint:(startingPoint + min_shape), 0:min_shape]

        imgCropped = cv2.cvtColor(imgCropped, cv2.COLOR_GRAY2BGR)

        imgCropped = cv2.resize(imgCropped, dim)
        croppedImages.append(imgCropped)

print("--- Processing Image %s seconds ---\n" % (time.time() - startProc))
overallStartPrediction = time.time()
for squareImg in croppedImages:
    squareImg = squareImg / 255
    squareImg = np.expand_dims(squareImg, axis=0)
    start_time_prediction = time.time()
    pred = model.predict(squareImg)
    #print("--- Pred single Image %s seconds ---\n" % (time.time() - start_time_prediction))
    #print('prediction: \n ')
    #print(pred)
    #print('\n')


print("--- Pred all Images %s seconds ---\n" % (time.time() - overallStartPrediction))