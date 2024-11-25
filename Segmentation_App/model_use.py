#from PIL import Image
# import os
# import pandas as pd
import numpy as np
import keras


def start(img):
    model = keras.models.load_model('models/mega_last_train(90+).h5')
    pixel_array = img.astype(np.float32)
    pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array))
    pixel_array = np.expand_dims(pixel_array, axis=0)
    prediction = model.predict(pixel_array)
    processed_prediction = np.squeeze(prediction)
    return processed_prediction


# def path():
#     return f'DICOM_anon'
#
#

#
#
# def buffer_imgs(filenames, folder='buffer'):
#     files = []
#     if not os.path.exists(folder):
#         os.makedirs(folder)
#     for filename in filenames:
#         img, header = load(filename)
#         pil = Image.fromarray(img.squeeze())
#         fname = folder + '/' + filename.replace('/', '-')
#         fname = fname+'.tiff'
#         pil.save(fname, 'TIFF', compression='none')
#         files.append(fname)
#     return pd.DataFrame(files)
#
#
# # def show_img(n):
# #     plt.figure(figsize=(5, 5))
# #     plt.imshow(Image.open(X[0][n]))
# #     plt.title(n)
#
#
# X_filenames = get_data()
# X_filenames = [path.replace('\\', '/') for path in X_filenames]
# print(f"Найдено {len(X_filenames)} файлов .dcm.")
# print(X_filenames)
#
# X = buffer_imgs(X_filenames)
# print(X.shape)
#
# # for i in np.random.choice(np.arange(X.shape[0]), 10):
# #     show_img(i)
#


