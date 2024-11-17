from PIL import Image
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, cm

from medpy.metric import dc, jc
from medpy.io import load, header

from sklearn.model_selection import train_test_split as tts

import tensorflow
import keras

SEED = 42


def path():
    return f'DICOM_anon'


def get_data():
    x_path = path()
    x_filenames = []
    if os.path.isdir(x_path):
        for filename in os.listdir(x_path):
            filepath = os.path.join(x_path, filename)
            x_filenames.append(filepath)
    else:
        print(f"Директория {x_path} не существует")
    return sorted(x_filenames)


def buffer_imgs(filenames, folder='buffer'):
    files = []
    if not os.path.exists(folder):
        os.makedirs(folder)
    for filename in filenames:
        img, header = load(filename)
        pil = Image.fromarray(img.squeeze())
        fname = folder + '/' + filename.replace('/', '-')
        fname = fname+'.tiff'
        pil.save(fname, 'TIFF', compression='none')
        files.append(fname)
    return pd.DataFrame(files)


# def show_img(n):
#     plt.figure(figsize=(5, 5))
#     plt.imshow(Image.open(X[0][n]))
#     plt.title(n)


X_filenames = get_data()
X_filenames = [path.replace('\\', '/') for path in X_filenames]
print(f"Найдено {len(X_filenames)} файлов .dcm.")
print(X_filenames)

X = buffer_imgs(X_filenames)
print(X.shape)

# for i in np.random.choice(np.arange(X.shape[0]), 10):
#     show_img(i)

# model = keras.models.load_model('models/last_train.h5')

