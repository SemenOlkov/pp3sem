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
import shutil
import threading


class CustomTheme(gr.Theme):
    def __init__(self):
        super().__init__()
        # self.body_background_fill = "linear-gradient(to right, #f8f9fa, #e9ecef)"
        self.text_color = "#7D7D7D"
        self.button_primary_background_fill = "white"
        self.button_secondary_background_fill = "#19A987"
        self.button_primary_text_color = "#0E8E70"
        self.button_secondary_text_color = "white"
        self.button_border_width = "1.5px"
        self.button_primary_border_color = "black"
        self.button_secondary_border_color = "#000000"
        self.button_primary_border_color_hover = "black"
        self.button_secondary_border_color_hover = "#000000"
        self.button_primary_background_fill_hover = "#00EBB4"
        self.button_secondary_background_fill_hover = "#0E8E70"
        self.button_primary_text_color_hover = "white"
        self.button_secondary_text_color_hover = "#FFFFFF"
        self.button_large_radius = "12px"
        self.input_background_fill = "#0E8E70"
        self.input_border_color = "#0E8E70"


theme = CustomTheme()


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

        return '<div class="markdown">Изображение отправлено на исправление. Спасибо за вашу помощь.</div>'
    return '<div class="markdown">Не выбрано изображение или не прикреплён DICOM</div>'


def process_dicom(image):
    dcm_file = pydicom.dcmread(image.name)
    before_segmentation = dcm_file.pixel_array
    main_fig, axis = plt.subplots()
    axis.axis('off')
    axis.imshow(before_segmentation.squeeze(), cmap='gray')
    if before_segmentation.ndim == 2:
        before_segmentation = np.expand_dims(before_segmentation, axis=0)
        before_segmentation = np.expand_dims(before_segmentation, axis=-1)
    before_segmentation = np.expand_dims(before_segmentation, axis=0)
    after_segmentation = model_use.start(before_segmentation)
    pred_8uc1 = (after_segmentation.squeeze() * 255).astype(np.uint8)
    contours_pred, _ = findContours(pred_8uc1, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)

    back_fig, back_axis = plt.subplots()
    back_axis.axis('off')
    back_axis.imshow(before_segmentation.squeeze(), cmap='gray')

    # segment_fig, segment_axis = plt.subplots()
    # segment_axis.imshow(after_segmentation.squeeze(), alpha=0.5, cmap='autumn')
    # axis.imshow(after_segmentation.squeeze(), alpha=0.5, cmap='autumn')
    for contour in contours_pred:
        axis.plot(contour[:, 0, 0], contour[:, 0, 1], 'r', linewidth=2)
        # segment_axis.plot(contour[:, 0, 0], contour[:, 0, 1], 'r', linewidth=2)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        main_fig.savefig(tmp_file.name, bbox_inches='tight')
        image_path = tmp_file.name

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        back_fig.savefig(tmp_file.name, bbox_inches='tight')
        back_path = tmp_file.name

    plt.close(main_fig)
    plt.close(back_fig)

    return image_path, back_path, image.name


with gr.Blocks(theme=theme, css_paths='static/styles/styles.css') as iface:
    processed_image_state = gr.State(value=None)
    dcm_path_state = gr.State(value=None)
    back_path_state = gr.State(value=None)
    with gr.Tab("Обводка печени на снимках компьютерной томографии", elem_classes="tabitem"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown('<div class="markdown">Окно с загруженным снимком</div>')
                dcm_input = gr.File(file_types=['.dcm'], elem_classes='input')
                process_button = gr.Button("Отправить снимок", variant='primary')

            with gr.Column(scale=3):
                gr.Markdown('<div class="markdown">Окно с обработанным снимком</div>')
                plot_output = gr.Image(elem_classes='output')
                file_output = gr.File(visible=False)


        def update_file_output(image):
            image_path, back_path, dcm_path = process_dicom(image)
            return image_path, gr.update(value=back_path), image_path, dcm_path, back_path  # Возвращаем back_path

        process_button.click(update_file_output, inputs=dcm_input, outputs=[plot_output, file_output, processed_image_state, dcm_path_state, back_path_state])

    with gr.Tab("Отправить исправление", elem_classes="tabitem"):
        gr.Markdown('<div class="markdown">Если обводка оказалась некорректной, вы можете самостоятельно обвести контур здесь, а также помочь нам усовершенствовать модель отправив исправление. Пожалуйста, если вы отправляете исправление, закрасьте ВСЮ печень на снимке ниже любым цветом кроме чёрного.</div>')
        with gr.Column():
            correction_input = gr.File(label="Загрузите изображение для исправления", file_types=['image'],
                                       visible=False)
            correction_image = gr.ImageEditor(label="Последнее отправленное изображение", interactive=True, sources=[], elem_classes='output', layers=False, show_fullscreen_button=True)
            edited_image_input = gr.File(visible=False)
            original_image_input = gr.File(visible=False)
            correction_output = gr.Markdown('', visible=False)
            download_output = gr.File(label="Скачать измененное изображение", file_types=['.png'], visible=False)
            with gr.Row():
                # send_image_button = gr.Button("Применить изменения", variant='primary')
                save_button = gr.Button("Отправить исправление", variant='primary')
                with gr.Column():
                    download_button = gr.Button("Скачать изображение", variant='secondary')
                    download_output = gr.File(visible=False)


        def update_correction_image(image):
            return image


        def update_correction_output(edited_image, dcm_path):
            if edited_image and isinstance(edited_image,
                                           dict) and 'layers' in edited_image and 'composite' in edited_image:
                composite = np.array(edited_image['composite'])
                layers = np.array(edited_image['layers'][0])
                black_background = np.zeros_like(composite, dtype=np.uint8)
                black_background[:] = [0, 0, 0, 255]
                masked_image = np.where(layers == [0, 0, 0, 0], black_background, composite).astype(np.uint8)
                white_color = [255, 255, 255, 255]
                masked_image[np.any(masked_image != [0, 0, 0, 255], axis=-1)] = white_color

                try:
                    pil_image = Image.fromarray(masked_image)
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                        pil_image.save(tmp_file.name)
                        output_message = save_correction_image(tmp_file.name, dcm_path)
                        return gr.update(value=output_message, visible=True)
                except Exception as e:
                    print(f"Ошибка при обработке изображения: {e}")
                    return None
            return None


        def update_correction_input(image_path):
            if image_path:
                return gr.update(value=image_path)


        def send_download_link(edited_image):
            if edited_image and isinstance(edited_image,
                                           dict) and 'layers' in edited_image and 'composite' in edited_image:
                edited_image = np.array(edited_image['composite'])
                try:
                    pil_image = Image.fromarray(edited_image)
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                        pil_image.save(tmp_file.name)
                        return gr.update(value=tmp_file.name, visible=True)
                except Exception as e:
                    print(f"Ошибка при загрузке изображения: {e}")
                    return None
            return None

        file_output.change(update_correction_input, inputs=file_output, outputs=correction_input)

        correction_input.change(update_correction_image, inputs=correction_input, outputs=correction_image)
        # send_image_button.click(update_edited_image_input, inputs=correction_image, outputs=edited_image_input)
        save_button.click(update_correction_output, inputs=[correction_image, dcm_path_state], outputs=correction_output)
        download_button.click(send_download_link, inputs=correction_image, outputs=download_output)


def create_download_interface():
    with gr.Blocks(title="Скачать corrections", theme=theme) as download_iface:
        download_adm_button = gr.Button("Скачать папку corrections", variant="primary")
        status_output = gr.Textbox(label="Статус:", visible=False)
        download_adm_output = gr.File(label="Файл для скачивания", visible=False)

        def create_archive():
            if not os.path.exists('corrections'):
                return gr.update(value="Папка corrections не найдена.", visible=True), gr.update(visible=False)

            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
                shutil.make_archive(tmp_file.name[:-4], 'zip', 'corrections')
                return gr.update(value="Скачайте отправленные исправления по ссылке ниже.", visible=True), gr.update(value=tmp_file.name, visible=True)

        download_adm_button.click(create_archive, inputs=[], outputs=[status_output, download_adm_output])
    return download_iface


def launch_gradio_app(app, port, share=False):
    app.launch(server_port=port, share=share)


download_interface = create_download_interface()
if __name__ == '__main__':
    threading.Thread(target=launch_gradio_app, args=(iface, 7860, True), daemon=True).start()
    threading.Thread(target=launch_gradio_app, args=(download_interface, 7861), daemon=True).start()
    threading.Event().wait()
