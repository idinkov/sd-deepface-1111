import os
import sys

# Get the absolute path of the directory containing the importing file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the relative path to the importing file to the system path
sys.path.append(os.path.join(current_dir, '../repo/dflive/'))

import cv2
from PIL import Image
import numpy as np
import gradio as gr
from pathlib import Path

from modules import processing, images
from modules import scripts, script_callbacks, shared, devices, modelloader
from modules.processing import Processed, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img
from modules.shared import opts, cmd_opts, state
from modules.sd_models import model_hash
from modules.paths import models_path
from basicsr.utils.download_util import load_file_from_url
from modules.ui import create_refresh_button
from scripts.ddetailerutils import DetectionDetailerScript, preload_ddetailer_model

from scripts.command import execute_deep_face_live_multiple
dd_models_path = os.path.join(models_path, "mmdet")

from scripts.dflutils import DflOptions, DflFiles
dfl_options = DflOptions(opts)

def list_models(include_downloadable=True):
    dfl_dropdown = []
    dfl_dropdown.append("None")
    #dfl_dropdown.append("Automatic")
    dfl_list = dfl_options.get_dfl_list(include_downloadable)
    for dfl in dfl_list:
        dfl_dropdown.append(dfl[0])
    return dfl_dropdown

def download_url(url, filename):
    load_file_from_url(url, filename)

from urllib.parse import urlparse

def is_url(string):
    try:
        result = urlparse(string)
        # Check if the URL has a scheme (e.g., http or https) and a network location (e.g., www.example.com)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def get_model_url(model_name):
    dfl_list = dfl_options.get_dfl_list(True)
    for dfl in dfl_list:
        if dfl[0] == model_name and is_url(dfl[1]):
            return dfl[1]
    return None

def list_detectors():
    return ['CenterFace', 'S3FD', 'YoloV5']

def list_markers():
    return ['OpenCV LBF', 'Google FaceMesh', 'InsightFace_2D106']

def list_align_modes():
    return ['From rectangle', 'From points', 'From static rect']

def startup():
    from launch import is_installed, run
    if not is_installed("mmdet"):
        python = sys.executable
        run(f'"{python}" -m pip install -U openmim==0.3.7', desc="Installing openmim", errdesc="Couldn't install openmim")
        run(f'"{python}" -m mim install mmcv-full==1.7.1', desc=f"Installing mmcv-full", errdesc=f"Couldn't install mmcv-full")
        run(f'"{python}" -m pip install mmdet==2.28.2', desc=f"Installing mmdet", errdesc=f"Couldn't install mmdet")

    if (DflFiles.folder_exists(dd_models_path) == False):
        print("No detection models found, downloading...")
        bbox_path = os.path.join(dd_models_path, "bbox")
        load_file_from_url("https://huggingface.co/dustysys/ddetailer/resolve/main/mmdet/bbox/mmdet_anime-face_yolov3.pth", bbox_path)
        load_file_from_url("https://huggingface.co/dustysys/ddetailer/raw/main/mmdet/bbox/mmdet_anime-face_yolov3.py", bbox_path)


startup()

def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


class DeepFaceLive(scripts.Script):
    def title(self):
        return "DeepFaceLive - AI Face Swap/Recovery"

    def show(self, is_img2img):
        return True


    def ui(self, is_img2img):
        import modules.ui

        gr.HTML("")
        with gr.Group():
            with gr.Row():
                gr.HTML("<p style=\"padding-left: 5px;\">You can put the models into <b>" + str(dfl_options.dfl_path) + "</b></p><br />")
            with gr.Row():
                dfm_model_dropdown = gr.Dropdown(label="DFM Model", choices=list_models(True), value="None", visible=True, type="value")
                create_refresh_button(dfm_model_dropdown, lambda: None, lambda: {"choices": list_models(True)}, "refresh_dfm_model_list")
                

            with gr.Row():
                image_return_original_checkbox = gr.Checkbox(label="Return original image")
                enable_detection_detailer_face_checkbox = gr.Checkbox(label="Enable Detection Detailer for face", value=False)
                save_detection_detailer_image_checkbox = gr.Checkbox(label="Return Detection Detailer Image")

        with gr.Group(elem_id="dfl_settings"):
            with gr.Tab("Face Detector"):
                gr.HTML("<br />")
                with gr.Row():
                    step_1_detector_input = gr.Dropdown(label="Detector", choices=list_detectors(), value="YoloV5", visible=True, type="value")
                    step_1_window_size_input = gr.Dropdown(label="Window size", choices=["Auto", "512", "320", "160"], value="Auto", type="value")
                    step_1_threshold_input = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="Threshold ", value=0.5)
                with gr.Row():
                    step_1_max_faces_input = gr.Number(label="Max Faces", min_value=1, max_value=None, value=3)
                    step_1_sort_by_input = gr.Dropdown(label="Sort By", choices=["Largest", "Dist from center", "From left to right", "From top to bottom", "From bottom to top"], value="Largest")

            with gr.Tab("Marker"):
                gr.HTML("<br />")
                with gr.Row():
                    step_2_marker_input = gr.Dropdown(label="Marker", choices=list_markers(), value="InsightFace_2D106", visible=True, type="value")
                    step_2_marker_coverage_input = gr.Slider(minimum=0.0, maximum=3.0, step=0.1, label="Marker coverage", value=1.6)

            with gr.Tab("Aligner"):
                gr.HTML("<br />")
                with gr.Row():
                    step_3_align_mode_input = gr.Dropdown(label="Align mode", choices=list_align_modes(), value="From points", visible=True, type="value")
                    step_3_face_coverage_input = gr.Slider(label="Face coverage ", minimum=0.0, maximum=3.0, step=0.1, value=2.2)
                    step_3_resolution_input = gr.Slider(label="Resolution ", minimum=16, maximum=640, step=16, value=320)
                with gr.Row():
                    step_3_exclude_moving_parts_input = gr.Checkbox(label="Exclude moving parts", value=True)
                    step_3_head_mode_input = gr.Checkbox(label="Head mode", value=False)
                    step_3_freeze_z_rotation_input = gr.Checkbox(label="Freeze Z rotation", value=False)
                with gr.Row():
                    step_3_x_offset_input = gr.Number(label="X offset", min_value=0, max_value=200, step=0.1, value=0)
                    step_3_y_offset_input = gr.Number(label="Y offset", min_value=0, max_value=200, step=0.1, value=0)

            with gr.Tab("Swapper"):
                gr.HTML("<br />")
                with gr.Row():
                    step_4_swap_all_faces_input = gr.Checkbox(label="Swap all faces", value=True)
                    step_4_face_id_input = gr.Number(label="Face ID", min_value=0, max_value=24, step=1, value=0)
                    step_4_two_pass_input = gr.Checkbox(label="Two Pass", value=False)
                    step_4_pre_sharpen_input = gr.Slider(label="Pre-sharpen ", minimum=0.0, maximum=8.0, step=0.1, value=4.0)
                with gr.Row():
                    step_4_pre_gamma_red_input = gr.Number(label="Pre-gamma Red", min_value=1, max_value=100, step=1, value=1)
                    step_4_pre_gamma_green_input = gr.Number(label="Pre-gamma Green", min_value=1, max_value=100, step=1, value=1)
                    step_4_pre_gamma_blue_input = gr.Number(label="Pre-gamma Blue", min_value=1, max_value=100, step=1, value=1)
                with gr.Row():
                    step_4_post_gamma_red_input = gr.Number(label="Post-gamma Red", min_value=1, max_value=100, step=1, value=1)
                    step_4_post_gamma_green_input = gr.Number(label="Post-gamma Green", min_value=1, max_value=100, step=1, value=1)
                    step_4_post_gamma_blue_input = gr.Number(label="Post-gamma Blue", min_value=1, max_value=100, step=1, value=1)


            with gr.Tab("Merger"):
                gr.HTML("<br />")
                with gr.Row():
                    step_5_x_offset_input = gr.Slider(label="Face X offset ", minimum=0, maximum=10, step=1, value=0)
                    step_5_y_offset_input = gr.Slider(label="Face Y offset ", minimum=0, maximum=10, step=1, value=0)
                    step_5_face_scale_input = gr.Slider(label="Face scale ", minimum=0.0, maximum=2.0, step=0.1, value=1.0)
                with gr.Row():
                    step_5_face_mask_src_input = gr.Checkbox(label="Face mask (SRC)", value=True)
                    step_5_face_celeb_input = gr.Checkbox(label="Face mask (CELEB)", value=True)
                    step_5_face_lmrks_input = gr.Checkbox(label="Face mask (LMRKS)", value=False)
                with gr.Row():
                    step_5_face_mask_erode_input = gr.Number(label="Face mask erode", min_value=0, max_value=100, step=1, value=5)
                    step_5_face_mask_blur_input = gr.Number(label="Face mask blur", min_value=0, max_value=100, step=1, value=25)
                    step_5_color_transfer_input = gr.Dropdown(label="Color transfer", choices=["none", "rct"], value="rct", visible=True, type="value")
                with gr.Row():
                    step_5_color_compression_input = gr.Slider(label="Color compression ", minimum=0, maximum=100, step=1, value=0)
                    step_5_face_opacity_input = gr.Slider(label="Face opacity ", minimum=0.00, maximum=1.00, step=0.01, value=1.00)
                    #step_5_interpolation_input = gr.Dropdown(label="Interpolation", choices=["bilinear", "bicubic", "lanczos4"], value="bilinear", visible=True, type="value")
                    step_5_interpolation_input = gr.Dropdown(label="Interpolation", choices=["bilinear"], value="bilinear", visible=True, type="value")

        return [dfm_model_dropdown,
                image_return_original_checkbox,
                enable_detection_detailer_face_checkbox,
                save_detection_detailer_image_checkbox,
                step_1_detector_input,
                step_1_window_size_input,
                step_1_threshold_input,
                step_1_max_faces_input,
                step_1_sort_by_input,
                step_2_marker_input,
                step_2_marker_coverage_input,
                step_3_align_mode_input,
                step_3_face_coverage_input,
                step_3_resolution_input,
                step_3_exclude_moving_parts_input,
                step_3_head_mode_input,
                step_3_freeze_z_rotation_input,
                step_3_x_offset_input,
                step_3_y_offset_input,
                step_4_swap_all_faces_input,
                step_4_face_id_input,
                step_4_two_pass_input,
                step_4_pre_sharpen_input,
                step_4_pre_gamma_red_input,
                step_4_pre_gamma_green_input,
                step_4_pre_gamma_blue_input,
                step_4_post_gamma_red_input,
                step_4_post_gamma_green_input,
                step_4_post_gamma_blue_input,
                step_5_x_offset_input,
                step_5_y_offset_input,
                step_5_face_scale_input,
                step_5_face_mask_src_input,
                step_5_face_celeb_input,
                step_5_face_lmrks_input,
                step_5_face_mask_erode_input,
                step_5_face_mask_blur_input,
                step_5_color_transfer_input,
                step_5_interpolation_input,
                step_5_color_compression_input,
                step_5_face_opacity_input]

    def process_frames(self, images, dfm_model_dropdown, step_1_detector_input,
                      step_1_window_size_input,
                      step_1_threshold_input,
                      step_1_max_faces_input,
                      step_1_sort_by_input,
                      step_2_marker_input,
                      step_2_marker_coverage_input,
                      step_3_align_mode_input,
                      step_3_face_coverage_input,
                      step_3_resolution_input,
                      step_3_exclude_moving_parts_input,
                      step_3_head_mode_input,
                      step_3_freeze_z_rotation_input,
                      step_3_x_offset_input,
                      step_3_y_offset_input,
                      step_4_swap_all_faces_input,
                      step_4_face_id_input,
                      step_4_two_pass_input,
                      step_4_pre_sharpen_input,
                      step_4_pre_gamma_red_input,
                      step_4_pre_gamma_green_input,
                      step_4_pre_gamma_blue_input,
                      step_4_post_gamma_red_input,
                      step_4_post_gamma_green_input,
                      step_4_post_gamma_blue_input,
                      step_5_x_offset_input,
                      step_5_y_offset_input,
                      step_5_face_scale_input,
                      step_5_face_mask_src_input,
                      step_5_face_celeb_input,
                      step_5_face_lmrks_input,
                      step_5_face_mask_erode_input,
                      step_5_face_mask_blur_input,
                      step_5_color_transfer_input,
                      step_5_interpolation_input,
                      step_5_color_compression_input,
                      step_5_face_opacity_input
                      ):

        from scripts.command import DetectorType, FaceSortBy, MarkerType, AlignMode
        print("Processing " + dfm_model_dropdown)
        img_array = []
        for orig_image in images:
            img_array.append(np.array(orig_image))

        dfm_path = str(Path(str(dfl_options.dfl_path) + "/" + str(dfm_model_dropdown)))

        step_1_detector = DetectorType.YOLOV5
        if step_1_detector_input == "CenterFace":
            step_1_detector = DetectorType.CENTER_FACE
        elif step_1_detector_input == "S3FD":
            step_1_detector = DetectorType.S3FD

        step_1_sort_by = FaceSortBy.LARGEST
        if step_1_sort_by_input == "Dist from center":
            step_1_sort_by = FaceSortBy.DIST_FROM_CENTER
        if step_1_sort_by_input == "From left to right":
            step_1_sort_by = FaceSortBy.LEFT_RIGHT
        if step_1_sort_by_input == "From top to bottom":
            step_1_sort_by = FaceSortBy.TOP_BOTTOM
        if step_1_sort_by_input == "From bottom to top":
            step_1_sort_by = FaceSortBy.BOTTOM_TOP

        step_1_window_size = step_1_window_size_input
        if step_1_window_size == "Auto":
            step_1_window_size = 0

        step_1_threshold = str(step_1_threshold_input)

        step_2_marker = MarkerType.INSIGHT_2D106
        if step_2_marker_input == "OpenCV LBF":
            step_2_marker = MarkerType.OPENCV_LBF
        elif step_2_marker_input == "Google FaceMesh":
            step_2_marker = MarkerType.GOOGLE_FACEMESH

        step_3_align_mode = AlignMode.FROM_POINTS
        if step_3_align_mode_input == "From rectangle":
            step_3_align_mode = AlignMode.FROM_RECT
        elif step_3_align_mode_input == "From static rect":
            step_3_align_mode = AlignMode.FROM_STATIC_RECT

        return execute_deep_face_live_multiple(numpy_images=img_array,
                                     dfm_path=dfm_path,
                                     device_id=0,
                                     step_1_detector=step_1_detector,
                                     step_1_window_size=step_1_window_size,
                                     step_1_threshold=step_1_threshold,
                                     step_1_max_faces=step_1_max_faces_input,
                                     step_1_sort_by=step_1_sort_by,
                                     step_2_marker=step_2_marker,
                                     step_2_marker_coverage=step_2_marker_coverage_input,
                                     step_3_align_mode=step_3_align_mode,
                                     step_3_face_coverage=step_3_face_coverage_input,
                                     step_3_resolution=int(step_3_resolution_input),
                                     step_3_exclude_moving_parts=step_3_exclude_moving_parts_input,
                                     step_3_head_mode=step_3_head_mode_input,
                                     step_3_freeze_z_rotation=step_3_freeze_z_rotation_input,
                                     step_3_x_offset=step_3_x_offset_input,
                                     step_3_y_offset=step_3_y_offset_input,
                                     step_4_swap_all_faces=step_4_swap_all_faces_input,
                                     step_4_face_id=step_4_face_id_input,
                                     step_4_two_pass=step_4_two_pass_input,
                                     step_4_pre_sharpen=step_4_pre_sharpen_input,
                                     step_4_pre_gamma_red=step_4_pre_gamma_red_input,
                                     step_4_pre_gamma_green=step_4_pre_gamma_green_input,
                                     step_4_pre_gamma_blue=step_4_pre_gamma_blue_input,
                                     step_4_post_gamma_red=step_4_post_gamma_red_input,
                                     step_4_post_gamma_green=step_4_post_gamma_green_input,
                                     step_4_post_gamma_blue=step_4_post_gamma_blue_input,
                                     step_5_x_offset=step_5_x_offset_input,
                                     step_5_y_offset=step_5_y_offset_input,
                                     step_5_face_scale=step_5_face_scale_input,
                                     step_5_face_mask_src=step_5_face_mask_src_input,
                                     step_5_face_celeb=step_5_face_celeb_input,
                                     step_5_face_lmrks=step_5_face_lmrks_input,
                                     step_5_face_mask_erode=step_5_face_mask_erode_input,
                                     step_5_face_mask_blur=step_5_face_mask_blur_input,
                                     step_5_color_transfer=step_5_color_transfer_input,
                                     step_5_interpolation=step_5_interpolation_input,
                                     step_5_color_compression=step_5_color_compression_input,
                                     step_5_face_opacity=step_5_face_opacity_input)

    def generate_batches(self, final_images, batch_size):
        """
        Generate batches of data based on a given batch size.

        Args:
            final_images (list): List of images to be batched.
            batch_size (int): The size of each batch.

        Returns:
            list: A list of batches, where each batch is a list of images.
        """
        num_batches = (len(final_images) + batch_size - 1) // batch_size
        batches = []
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(final_images))
            batches.append(final_images[start:end])
        return batches

    def run(self, p,
            dfm_model_dropdown,
            image_return_original_checkbox,
            enable_detection_detailer_face_checkbox,
            save_detection_detailer_image_checkbox,
            step_1_detector_input,
            step_1_window_size_input,
            step_1_threshold_input,
            step_1_max_faces_input,
            step_1_sort_by_input,
            step_2_marker_input,
            step_2_marker_coverage_input,
            step_3_align_mode_input,
            step_3_face_coverage_input,
            step_3_resolution_input,
            step_3_exclude_moving_parts_input,
            step_3_head_mode_input,
            step_3_freeze_z_rotation_input,
            step_3_x_offset_input,
            step_3_y_offset_input,
            step_4_swap_all_faces_input,
            step_4_face_id_input,
            step_4_two_pass_input,
            step_4_pre_sharpen_input,
            step_4_pre_gamma_red_input,
            step_4_pre_gamma_green_input,
            step_4_pre_gamma_blue_input,
            step_4_post_gamma_red_input,
            step_4_post_gamma_green_input,
            step_4_post_gamma_blue_input,
            step_5_x_offset_input,
            step_5_y_offset_input,
            step_5_face_scale_input,
            step_5_face_mask_src_input,
            step_5_face_celeb_input,
            step_5_face_lmrks_input,
            step_5_face_mask_erode_input,
            step_5_face_mask_blur_input,
            step_5_color_transfer_input,
            step_5_interpolation_input,
            step_5_color_compression_input,
            step_5_face_opacity_input
            ):
        processing.fix_seed(p)
        seed = p.seed
        initial_info = None
        p.do_not_save_grid = True
        p.do_not_save_samples = True
        is_txt2img = isinstance(p, StableDiffusionProcessingTxt2Img)
        p_txt = p
        print(step_1_threshold_input)
        if (not is_txt2img):
            orig_image = p.init_images[0]
        else:
            p = StableDiffusionProcessingImg2Img(
                batch_size=p_txt.batch_size,
                init_images=None,
                resize_mode=0,
                denoising_strength=0.4,
                mask=None,
                mask_blur=4,
                inpainting_fill=1,
                inpaint_full_res=1,
                inpaint_full_res_padding=32,
                inpainting_mask_invert=0,
                sd_model=p_txt.sd_model,
                outpath_samples=p_txt.outpath_samples,
                outpath_grids=p_txt.outpath_grids,
                prompt=p_txt.prompt,
                negative_prompt=p_txt.negative_prompt,
                styles=p_txt.styles,
                seed=p_txt.seed,
                subseed=p_txt.subseed,
                subseed_strength=p_txt.subseed_strength,
                seed_resize_from_h=p_txt.seed_resize_from_h,
                seed_resize_from_w=p_txt.seed_resize_from_w,
                sampler_name=p_txt.sampler_name,
                n_iter=p_txt.n_iter,
                steps=p_txt.steps,
                cfg_scale=p_txt.cfg_scale,
                width=p_txt.width,
                height=p_txt.height,
                tiling=p_txt.tiling,
            )
            p.do_not_save_grid = True
            p.do_not_save_samples = True

        import math
        output_images = []
        factor_jobs = 0
        add_job_count = 0
        batch_size_dfl = 40
        if dfm_model_dropdown != "None":
            if get_model_url(dfm_model_dropdown) is not None:
                add_job_count += 1
            add_job_count = math.ceil((p_txt.n_iter * p_txt.batch_size)/batch_size_dfl)
        if enable_detection_detailer_face_checkbox:
            factor_jobs += 1
        state.job_count = p_txt.n_iter + (p_txt.n_iter * p_txt.batch_size)*factor_jobs + add_job_count
        processed = processing.process_images(p_txt)
        if initial_info is None:
            initial_info = processed.info
        final_images = []
        images_count = len(processed.images)

        if enable_detection_detailer_face_checkbox:
            ddetailer_model = preload_ddetailer_model("bbox/mmdet_anime-face_yolov3.pth")

        model_location = dfm_model_dropdown
        print(dfm_model_dropdown)
        print(get_model_url(dfm_model_dropdown))
        if get_model_url(dfm_model_dropdown) is not None:
            download_url_name = get_model_url(dfm_model_dropdown)
            download_location = str(dfl_options.get_dfl_path()) + "/" + os.path.basename(download_url_name)
            state.job = f"Downloading model " + dfm_model_dropdown + " from " + download_url_name + " to " + download_location
            download_url(download_url_name, str(dfl_options.get_dfl_path()))
            model_location = download_url_name.split("/")[-1]
            state.job_no += 1

        for image_n, current_image in enumerate(processed.images):
            text_generation = f"Generation {(image_n+1)} out of {images_count}"
            print(text_generation)
            state.job = text_generation
            devices.torch_gc()
            if is_txt2img:
                init_image = current_image
            else:
                init_image = orig_image

            if image_return_original_checkbox or (dfm_model_dropdown == "None" and not enable_detection_detailer_face_checkbox):
                output_images.append(init_image)

            if enable_detection_detailer_face_checkbox:
                ddscript = DetectionDetailerScript()
                last_no = state.job_no
                p.prompt = p_txt.prompt
                init_image = ddscript.run(p=p, model=ddetailer_model, model_name="bbox/mmdet_anime-face_yolov3.pth", init_image=init_image)
                if last_no == state.job_no:
                    state.job_no += 1
                if save_detection_detailer_image_checkbox or dfm_model_dropdown == "None":
                    output_images.append(init_image)

            final_images.append(init_image)

        if dfm_model_dropdown != "None":
            batches = self.generate_batches(final_images, batch_size_dfl)
            for batch in batches:
                output_images_itteration = self.process_frames(images=batch,
                                    dfm_model_dropdown=model_location,
                                    step_1_detector_input=step_1_detector_input,
                                    step_1_window_size_input=step_1_window_size_input,
                                    step_1_threshold_input=step_1_threshold_input,
                                    step_1_max_faces_input=step_1_max_faces_input,
                                    step_1_sort_by_input=step_1_sort_by_input,
                                    step_2_marker_input=step_2_marker_input,
                                    step_2_marker_coverage_input=step_2_marker_coverage_input,
                                    step_3_align_mode_input=step_3_align_mode_input,
                                    step_3_face_coverage_input=step_3_face_coverage_input,
                                    step_3_resolution_input=step_3_resolution_input,
                                    step_3_exclude_moving_parts_input=step_3_exclude_moving_parts_input,
                                    step_3_head_mode_input=step_3_head_mode_input,
                                    step_3_freeze_z_rotation_input=step_3_freeze_z_rotation_input,
                                    step_3_x_offset_input=step_3_x_offset_input,
                                    step_3_y_offset_input=step_3_y_offset_input,
                                    step_4_swap_all_faces_input=step_4_swap_all_faces_input,
                                    step_4_face_id_input=step_4_face_id_input,
                                    step_4_two_pass_input=step_4_two_pass_input,
                                    step_4_pre_sharpen_input=step_4_pre_sharpen_input,
                                    step_4_pre_gamma_red_input=step_4_pre_gamma_red_input,
                                    step_4_pre_gamma_green_input=step_4_pre_gamma_green_input,
                                    step_4_pre_gamma_blue_input=step_4_pre_gamma_blue_input,
                                    step_4_post_gamma_red_input=step_4_post_gamma_red_input,
                                    step_4_post_gamma_green_input=step_4_post_gamma_green_input,
                                    step_4_post_gamma_blue_input=step_4_post_gamma_blue_input,
                                    step_5_x_offset_input=step_5_x_offset_input,
                                    step_5_y_offset_input=step_5_y_offset_input,
                                    step_5_face_scale_input=step_5_face_scale_input,
                                    step_5_face_mask_src_input=step_5_face_mask_src_input,
                                    step_5_face_celeb_input=step_5_face_celeb_input,
                                    step_5_face_lmrks_input=step_5_face_lmrks_input,
                                    step_5_face_mask_erode_input=step_5_face_mask_erode_input,
                                    step_5_face_mask_blur_input=step_5_face_mask_blur_input,
                                    step_5_color_transfer_input=step_5_color_transfer_input,
                                    step_5_interpolation_input=step_5_interpolation_input,
                                    step_5_color_compression_input=step_5_color_compression_input,
                                    step_5_face_opacity_input=step_5_face_opacity_input)

                for output_image in output_images_itteration:
                    output_images.append(output_image)

                state.job_no += 1

        if initial_info is None:
            initial_info = "No initial info"

        return Processed(p, output_images, seed, initial_info)



def get_device():
    device_id = shared.cmd_opts.device_id
    if device_id is not None:
        cuda_device = f"cuda:{device_id}"
    else:
        cuda_device = "cpu"
    return cuda_device