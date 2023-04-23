import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from notebook import config
import matplotlib.pyplot as plt
from dataloader.img_aug import img_aug
from utils.imgShow import imgShow, imsShow
from nets.watnet import watnet
from dataloader.tfrecord_io import parse_image, parse_shape, toPatchPair
os.chdir('..')

# data loading from .tfrecord file
path_tra_data_scene = 'data/tfrecord-s2/tra_scene.tfrecords'
path_val_data_scene = 'data/tfrecord-s2/val_scene.tfrecords'
path_val_data_patch = 'data/tfrecord-s2/val_patch.tfrecords'
# training data
tra_dset = tf.data.TFRecordDataset([path_tra_data_scene, path_val_data_scene])  # final training

tra_dset = tra_dset.map(parse_image).map(parse_shape)\
            .cache()\
            .map(toPatchPair)\
            .map(img_aug)
tra_dset = tra_dset.shuffle(config.buffer_size).batch(config.batch_size)

# validation data
val_dset = tf.data.TFRecordDataset(path_val_data_patch)
val_dset = val_dset.map(parse_image).map(parse_shape)\
            .map(toPatchPair)\
            .cache()

val_dset = val_dset.batch(16)

# check
# for i in range(5):
start = time.time()
i_batch = i_scene = 0
for patch, truth in tra_dset:
    i_batch += 1
    i_scene += patch.shape[0]
imsShow(img_list=[patch[0], truth[0]],
        img_name_list=['patch', 'truth'],
        clip_list=[2,0])

plt.show()
print('num of scenes:', i_scene)
print('time:', time.time()-start)

# model configuration
model = watnet(input_shape=(config.patch_size, config.patch_size, config.num_bands), nclasses=2)

'''------1. train step------'''


@tf.function
def train_step(model, loss_fun, optimizer, x, y):
    with tf.GradientTape() as tape:
        y_pre = model(x, training=True)
        loss = loss_fun(y, y_pre)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    config.tra_loss.update_state(loss)
    config.tra_oa.update_state(y, y_pre)
    config.tra_miou.update_state(y, y_pre)
    return config.tra_loss.result(), config.tra_oa.result(), config.tra_miou.result()


'''------2. test step------'''


@tf.function
def test_step(model, loss_fun, x, y):
    with tf.GradientTape() as tape:
        y_pre = model(x, training=False)
        loss = loss_fun(y, y_pre)
    config.val_loss.update_state(loss)
    config.val_oa.update_state(y, y_pre)
    config.val_miou.update_state(y, y_pre)
    return config.val_loss.result(), config.val_oa.result(), config.val_miou.result()


'''------3. train loops------'''


def train_loops(model, loss_fun, optimizer, tra_dset, val_dset, epochs):
    miou_plot, loss_plot = [], []
    for epoch in range(epochs):
        start = time.time()
        # --- train the model ---
        for x_batch, y_batch in tra_dset:
            tra_loss_epoch, tra_oa_epoch, tra_miou_epoch = train_step(model, loss_fun, optimizer, x_batch, y_batch)
        # --- test the model ---
        # if epoch == 0 or epoch>200:
        for x_batch, y_batch in val_dset:
            val_loss_epoch, val_oa_epoch, val_miou_epoch = test_step(model, loss_fun, x_batch, y_batch)
        # --- update the metrics ---
        config.tra_loss.reset_states(), config.tra_oa.reset_states(), config.tra_miou.reset_states()
        config.val_loss.reset_states(), config.val_oa.reset_states(), config.val_miou.reset_states()
        format = 'Ep {}/{}: traLoss:{:.3f},traOA:{:.3f},traMIoU:{:.3f},valLoss:{:.3f},valOA:{:.3f},valMIoU:{:.3f},time:{:.1f}s'
        print(format.format(epoch + 1, config.epochs, tra_loss_epoch, tra_oa_epoch, tra_miou_epoch, val_loss_epoch,
                            val_oa_epoch, val_miou_epoch, time.time() - start))
        miou_plot.append(val_miou_epoch.numpy())
        loss_plot.append(val_loss_epoch.numpy())
        # --- visualize the results ---
        if epoch % 10 == 0:
            i = np.random.randint(16)
            for val_patch, val_truth in val_dset.take(1):
                plt.figure(figsize=(10, 4))
                pre = model(val_patch, training=False)
                imsShow(img_list=[val_patch.numpy()[i], val_truth.numpy()[i], pre.numpy()[i]], \
                        img_name_list=['val_patch', 'val_truth', 'prediction'], \
                        clip_list=[2, 0, 0], \
                        color_bands_list=None)
                plt.show()

    return miou_plot, loss_plot

# training
with tf.device('/device:GPU:0'):
    miou_plot, loss_plot =train_loops(model=model, \
                        loss_fun=config.loss_bce, \
                        optimizer=config.opt_adam, \
                        tra_dset=tra_dset, \
                        val_dset=val_dset, \
                        epochs=config.epochs,
                        )

# saving metric
metric_path = "result/metrics_watnet.csv"
dataframe = pd.DataFrame({'val_miou':miou_plot, 'val_loss':loss_plot})
dataframe.to_csv(metric_path, index=False, sep=',')

# model saving.h5
path_save = 'model/watnet.h5'
model.save(path_save)
