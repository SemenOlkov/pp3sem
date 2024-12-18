import gradio as gr
import pydicom
import matplotlib.pyplot as plt
import model_use
from cv2 import CHAIN_APPROX_SIMPLE, RETR_EXTERNAL, findContours
import numpy as np
import tempfile
import os
from PIL import Image
from datetime import datetime


def save_correction_image(image, dcm_path):
    if image and dcm_path:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        image_extension = os.path.splitext(image)[1]
        save_image_path = f'corrections/ground/correction_ground_{timestamp}{image_extension}'
        save_dcm_path = f'corrections/DICOM/dcm_{timestamp}.dcm'

        os.makedirs('corrections/ground', exist_ok=True)
        os.makedirs('corrections/DICOM', exist_ok=True)

        # Save image
        with open(image, 'rb') as file_data:
            with open(save_image_path, 'wb') as file:
                file.write(file_data.read())

        # Save DICOM
        with open(dcm_path, 'rb') as dcm_file:
            with open(save_dcm_path, 'wb') as file:
                file.write(dcm_file.read())

        return "Изображение отправлено на исправление."
    return "Не выбрано изображение или не прикреплён DICOM"


def process_dicom(image):
    dcm_file = pydicom.dcmread(image.name)
    before_segmentation = dcm_file.pixel_array
    after_segmentation = model_use.start(before_segmentation)
    pred_8uc1 = (after_segmentation.squeeze() * 255).astype(np.uint8)
    contours_pred, _ = findContours(pred_8uc1, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    fig, ax2 = plt.subplots()
    ax2.axis('off')
    ax2.imshow(before_segmentation.squeeze(), cmap='gray')
    ax2.imshow(after_segmentation.squeeze(), alpha=0.5, cmap='autumn')
    for contour in contours_pred:
        ax2.plot(contour[:, 0, 0], contour[:, 0, 1], 'r', linewidth=2)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        fig.savefig(tmp_file.name, bbox_inches='tight')
        image_path = tmp_file.name

    plt.close(fig)
    return image_path, image.name


with gr.Blocks() as iface:
    processed_image_state = gr.State(value=None)
    dcm_path_state = gr.State(value=None)
    with gr.Tab("Обводка печени на снимках компьютерной томографии"):
        with gr.Row():
            with gr.Column(scale=1):
                dcm_input = gr.File(file_types=['.dcm'])
                process_button = gr.Button("Отправить снимок", variant="primary")

            with gr.Column(scale=3):
                plot_output = gr.Image()
                file_output = gr.File(visible=False)


        def update_file_output(image):
            image_path, dcm_path = process_dicom(image)
            return image_path, gr.update(value=image_path, visible=True), image_path, dcm_path


        process_button.click(update_file_output, inputs=dcm_input,
                             outputs=[plot_output, file_output, processed_image_state, dcm_path_state])

    with gr.Tab("Отправить исправление"):
        with gr.Column():
            correction_input = gr.File(label="Загрузите изображение для исправления", file_types=['image'],
                                       visible=False)
            correction_image = gr.ImageEditor(label="Последнее обработанное изображение", interactive=True,
                                              height='40%')
            edited_image_input = gr.File(visible=False)  # Добавленный скрытый инпут
            correction_output = gr.Textbox(label="Статус:", visible=False)
            with gr.Row():
                send_image_button = gr.Button("Применить изменения",
                                              variant="secondary")  # Кнопка для отправки отредактированного изображения
                save_button = gr.Button("Отправить исправление", variant='primary')


        def update_correction_image(image):
            return image


        def update_correction_output(image, dcm_path):
            output_message = save_correction_image(image, dcm_path)
            return gr.update(value=output_message, visible=True)


        def update_correction_input(image_path):
            if image_path:
                return gr.update(value=image_path)


        def update_edited_image_input(edited_image):
            if edited_image and isinstance(edited_image, dict) and 'composite' in edited_image:
                composite_image = edited_image['composite']
                try:
                    pil_image = Image.fromarray(composite_image.astype(np.uint8))
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                        pil_image.save(tmp_file.name)
                        return tmp_file.name
                except Exception as e:
                    print(f"Ошибка при обработке изображения: {e}")
                    return None
            return None


        file_output.change(update_correction_input, inputs=file_output, outputs=correction_input)

        correction_input.change(update_correction_image, inputs=correction_input, outputs=correction_image)
        send_image_button.click(update_edited_image_input, inputs=correction_image,
                                outputs=edited_image_input)  # Обработчик нажатия кнопки
        save_button.click(update_correction_output, inputs=[edited_image_input, dcm_path_state],
                          outputs=correction_output)  # Используем скрытый инпут в функции

iface.launch()