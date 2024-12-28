import numpy as np
import tensorflow


def start(img):
    model = tensorflow.keras.models.load_model('static/models/curr_model.keras', custom_objects={'dice_coef_loss': 1, 'dice_coef': 0})
    img = img.astype('float32') / 255.0
    mean = np.mean(img)
    std = np.std(img)
    if std != 0:
        img = (img - mean) / std
    prediction = model.predict(img)
    prediction = (prediction * 255).astype('uint8')
    return prediction
