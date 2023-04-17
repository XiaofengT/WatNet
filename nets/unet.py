from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from nets.vgg import VGG16


def Unet(input_shape, num_classes=2):
    inputs = Input(input_shape)
    # -------------------------------#
    #   获得五个有效特征层
    #   feat1   128,128,16
    #   feat2   64,64,32
    #   feat3   32,32,64
    #   feat4   16,16,128
    #   feat5   8,8,128
    # -------------------------------#
    feat1, feat2, feat3, feat4, feat5 = VGG16(inputs)

    channels = [16, 32, 64, 128]
    # 8,8,128 -> 16,16,128
    P5_up = UpSampling2D(size=(2, 2))(feat5)
    # P4: 16,16,256
    P4 = Concatenate(axis=3)([feat4, P5_up])
    # 16,16,256 -> 16,16,128
    P4 = Conv2D(channels[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P4)
    P4 = Conv2D(channels[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P4)

    # 16,16,128 -> 32,32,128
    P4_up = UpSampling2D(size=(2, 2))(P4)
    # P3: 32,32,192
    P3 = Concatenate(axis=3)([feat3, P4_up])
    # 32,32,192 -> 32,32,64
    P3 = Conv2D(channels[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P3)
    P3 = Conv2D(channels[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P3)

    # 32,32,64 -> 64,64,64
    P3_up = UpSampling2D(size=(2, 2))(P3)
    # P2: 64,64,96
    P2 = Concatenate(axis=3)([feat2, P3_up])
    # 64,64,96 -> 64,64,32
    P2 = Conv2D(channels[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P2)
    P2 = Conv2D(channels[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P2)

    # 64,64,32 -> 128,128,32
    P2_up = UpSampling2D(size=(2, 2))(P2)
    # P1: 128,128,48
    P1 = Concatenate(axis=3)([feat1, P2_up])
    # 128,128,48 -> 128,128,16
    P1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P1)
    P1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P1)

    # 128,128,16 -> 128,128,num_classes
    P1 = Conv2D(num_classes, 1, activation="sigmoid")(P1)

    model = Model(inputs=inputs, outputs=P1)
    return model
