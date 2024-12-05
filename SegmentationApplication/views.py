from django.shortcuts import render
import pydicom
import matplotlib.pyplot as plt
from cv2 import CHAIN_APPROX_SIMPLE, RETR_EXTERNAL, findContours
import numpy as np
import keras
import tensorflow
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import ops as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization
import io
import base64
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt


tensorflow.keras.backend.set_image_data_format('channels_first')


def unet_1(img_channels, image_rows, image_cols, neurons=16):
    inputs = Input((img_channels, image_rows, image_cols))
    conv1 = Conv2D(neurons*1, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(neurons*1, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(neurons*2, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(neurons*2, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(neurons*4, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(neurons*4, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(neurons*8, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(neurons*8, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    conv5 = Conv2D(neurons*16, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(neurons*16, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    up6 = Dropout(0.5)(up6)
    conv6 = Conv2D(neurons*8, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(neurons*8, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    up7 = Dropout(0.5)(up7)
    conv7 = Conv2D(neurons*4, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(neurons*4, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    up8 = Dropout(0.5)(up8)
    conv8 = Conv2D(neurons*2, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(neurons*2, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    up9 = Dropout(0.5)(up9)
    conv9 = Conv2D(neurons*1, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(neurons*1, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Dropout(0.5)(conv9)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv10)

    model = Model(inputs=[inputs], outputs=[conv10])
    return model

@csrf_exempt
def upload_image(request):
    if request.method == 'POST':
        image = request.FILES.get('image')
        dcm_file = pydicom.dcmread(image)

        before_segmentation = dcm_file.pixel_array
        w_size = 512
        model = unet_1(1, w_size, w_size, 8)
        model.load_weights('models/mega_last_train(90+).h5')
        # a = tensorflow.keras.models.load_model('models/mega_last_train(90+).h5')
        pixel_array = before_segmentation.astype('uint8')
        pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array))
        pixel_array = np.expand_dims(pixel_array, axis=0)
        pixel_array = np.expand_dims(pixel_array, axis=1)
        after_segmentation = model.predict(pixel_array).astype('uint8')
        pred_8uc1 = (after_segmentation.squeeze() * 255).astype(np.uint8)
        contours_pred, _ = findContours(pred_8uc1, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(before_segmentation.squeeze(), cmap='gray')
        ax1.imshow(before_segmentation.squeeze(), alpha=0.5, cmap='autumn')
        ax2.imshow(before_segmentation.squeeze(), cmap='gray')
        ax2.imshow(after_segmentation.squeeze(), alpha=0.5, cmap='autumn')
        for contour in contours_pred:
            ax2.plot(contour[:, 0, 0], contour[:, 0, 1], 'r', linewidth=2)

        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        return JsonResponse({'image': image_base64})
    else:
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)


def index(request):
    return render(request, 'templates/index.html')

def upload_view(request):
    return render(request, 'upload.html')
