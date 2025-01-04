import numpy as np
import keras

def start(img):
    """
    Выполняет сегментацию изображения с использованием загруженной нейронной сети.

    Args:
        img (numpy.ndarray): Входное изображение в формате numpy array.
                             Ожидается, что изображение будет иметь форму (1, height, width, channels).

    Returns:
        numpy.ndarray: Прогнозируемая маска сегментации в формате numpy array,
                       где каждый пиксель представлен целым числом от 0 до 255 (uint8).
                       Маска имеет ту же форму, что и входное изображение, но без первой размерности (1).
    """
    # Загрузка предварительно обученной модели Keras.
    model = keras.models.load_model('static/models/curr_model.keras', custom_objects={'dice_coef_loss': 1, 'dice_coef': 0})

    # Приведение типа данных изображения к float32 и нормализация значений пикселей от 0 до 1.
    img = img.astype('float32') / 255.0

    # Вычисление среднего и стандартного отклонения пикселей изображения.
    mean = np.mean(img)
    std = np.std(img)

    # Стандартизация изображения, если стандартное отклонение не равно нулю.
    # Это помогает улучшить производительность модели.
    if std != 0:
        img = (img - mean) / std

    # Получение предсказания от модели.
    prediction = model.predict(img)

    # Масштабирование предсказания к диапазону [0, 255] и приведение типа данных к uint8.
    prediction = (prediction * 255).astype('uint8')

    return prediction
