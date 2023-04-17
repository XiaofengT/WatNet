from tensorflow.keras import layers


def VGG16(img_input):
    # Block 1
    # 128,128,3 -> 128,128,16
    x = layers.Conv2D(16, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(16, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    # 引出初步的有效特征层, 用于构建加强特征提取网络
    feat1 = x
    # 128,128,16 -> 64,64,16
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    # 64,64,16 -> 64,64,32
    x = layers.Conv2D(32, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(32, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    # 引出有效特征层, 用于构建加强特征提取网络
    feat2 = x
    # 64,64,32 -> 32,32,32
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    # 32,32,32 -> 32,32,64
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    # 引出有效特征层, 用于构建加强特征提取网络
    feat3 = x
    # 32,32,64 -> 16,16,64
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    # 16,16,64 -> 16,16,128
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    # 引出有效特征层, 用于构建加强特征提取网络
    feat4 = x
    # 16,16,128 -> 8,8,128
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    # 8,8,128 -> 8,8,128
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    # 引出有效特征层, 用于构建加强特征提取网络
    feat5 = x
    return feat1, feat2, feat3, feat4, feat5
