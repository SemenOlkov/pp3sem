import numpy as np
import tensorflow
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization
import keras
from keras import ops as K

def dice_coef(y_true, y_pred):
    smooth = 1e-20
    y_true_f = K.cast(y_true, 'float32')
    intersection = K.sum(y_true_f * y_pred)
    return (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def start(img):
    model = tensorflow.keras.models.load_model('models/qqq.keras', custom_objects={'dice_coef_loss': dice_coef_loss,
                                                                                   'dice_coef': dice_coef})
    img = img.astype('float32') / 255.0
    mean = np.mean(img)
    std = np.std(img)
    if std != 0:
        img = (img - mean) / std
    prediction = model.predict(img)
    prediction = (prediction * 255).astype('uint8')
    return prediction
