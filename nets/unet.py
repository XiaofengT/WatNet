from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from nets.vgg import VGG16


def Unet(input_shape, num_classes=2):
    inputs = Input(input_shape)
    feat1, feat2, feat3, feat4, feat5 = VGG16(inputs)

    channels = [64, 128, 256, 512]
    # 32,32,512 -> 64,64,512
    P5_up = UpSampling2D(size=(2, 2))(feat5)
    # P4: 64,64,1024
    P4 = Concatenate(axis=3)([feat4, P5_up])
    # 64,64,1024 -> 64,64,512
    P4 = Conv2D(channels[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P4)
    P4 = Conv2D(channels[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P4)

    # 64,64,512 -> 128,128,512
    P4_up = UpSampling2D(size=(2, 2))(P4)
    # P3: 128,128,768
    P3 = Concatenate(axis=3)([feat3, P4_up])
    # 128,128,768 -> 128,128,256
    P3 = Conv2D(channels[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P3)
    P3 = Conv2D(channels[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P3)

    # 128,128,256 -> 256,256,256
    P3_up = UpSampling2D(size=(2, 2))(P3)
    # P2: 256,256,384
    P2 = Concatenate(axis=3)([feat2, P3_up])
    # 256,256,384 -> 256,256,128
    P2 = Conv2D(channels[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P2)
    P2 = Conv2D(channels[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P2)

    # 256,256,128 -> 512,512,128
    P2_up = UpSampling2D(size=(2, 2))(P2)
    # P1: 512,512,192
    P1 = Concatenate(axis=3)([feat1, P2_up])
    # 512,512,192 -> 512,512,64
    P1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P1)
    P1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P1)

    # 512,512,64 -> 512,512,num_classes
    P1 = Conv2D(num_classes, 1, activation="sigmoid")(P1)

    model = Model(inputs=inputs, outputs=P1)
    return model
