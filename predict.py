import os
import numpy as np
from tqdm import tqdm
import cv2 as cv
import tensorflow as tf
from nets.unet import Unet
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

images = []
mask = []

image_path = 'D:/PYTHON/waterbody_segement/WatNet/Water_Bodies_Dataset/Images/'
mask_path = 'D:/PYTHON/waterbody_segement/WatNet/Water_Bodies_Dataset/Masks/'

image_names = sorted(next(os.walk(image_path))[-1])
mask_names = sorted(next(os.walk(image_path))[-1])

if image_names == mask_names:
    print('Image and Mask are correctly Placed')

SIZE = 128
images = np.zeros(shape=(len(image_names), SIZE, SIZE, 3))
masks = np.zeros(shape=(len(image_names), SIZE, SIZE, 1))

for id in tqdm(range(len(image_names)), desc="Images"):
    path = image_path + image_names[id]
    img = img_to_array(load_img(path)).astype('float') / 255.
    img = cv.resize(img, (SIZE, SIZE), cv.INTER_AREA)
    images[id] = img

for id in tqdm(range(len(mask_names)), desc="Mask"):
    path = mask_path + mask_names[id]
    mask = img_to_array(load_img(path)).astype('float') / 255.
    mask = cv.resize(mask, (SIZE, SIZE), cv.INTER_AREA)
    masks[id] = mask[:, :, :1]

inputs = [SIZE, SIZE, 3]
model = Unet(inputs, num_classes=1)
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint('model/UNet-01.h5', save_best_only=True),
]

model.summary()

X, y = images[:int(len(images)*0.9)], masks[:int(len(images)*0.9)]
test_X, test_y = images[int(len(images)*0.9):], masks[int(len(images)*0.9):]

with tf.device("/GPU:0"):
    results = model.fit(
        X, y,
        epochs=100,
        callbacks=callbacks,
        validation_split=0.1,
        batch_size=16
    )
