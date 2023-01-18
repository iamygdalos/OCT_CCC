from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt


base_model = keras.applications.ResNet50(
            weights='imagenet',  # Load weights pre-trained on ImageNet.
            input_shape=(224, 224, 3),
            include_top=False)  # Do not include the ImageNet classifier at the top.
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)

model.load_weights('D:\\MA_Luisa\\models\\single0\\ckpt\\resnet_MaskedSquared_0')

baseLayers = model.layers[1]
out2 = baseLayers.layers[2]

featureModel = keras.Model(base_model.inputs, base_model.layers[7].output)

img = cv2.imread('D:\\MA_Luisa\\data\\exportKRLM_Singles_MaskedSquared\\normal\\2006291850IA_B\\2\\2006291850IA_B_500_2_1.png', cv2.IMREAD_UNCHANGED)
img = cv2.resize(img, (224,224))
img = np.expand_dims(img, axis=0)

featureMaps = featureModel.predict(img)

square = 8
ix = 1
for x in range(square):
        for y in range(square):
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(featureMaps[0,:,:,ix-1], cmap='gray')
            ix += 1

plt.show()
