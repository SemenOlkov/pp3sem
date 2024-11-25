import tkinter as tk
from tkinter import filedialog
import pydicom
import matplotlib.pyplot as plt
import model_use

root = tk.Tk()
root.withdraw()  # Убираем главное окно

filepath = filedialog.askopenfilename(filetypes=[("DICOM files", "*.dcm")])

if filepath:
    try:
        dcm_file = pydicom.dcmread(filepath)

        before_segmentation = dcm_file.pixel_array
        # after_segmentation = model_use.start(before_segmentation)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(before_segmentation, cmap='autumn')
        axes[0].set_title("Загруженное изображение")

        axes[1].imshow(before_segmentation, cmap='autumn')
        axes[1].set_title("Результат обработки")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Произошла ошибка: {e}")