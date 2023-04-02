import os
import sys
sys.path.append('repo/dflive/')

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
from scripts.ddetailerutils import DetectionDetailerScript

from scripts.command import execute_deep_face_live
dd_models_path = os.path.join(models_path, "mmdet")

from scripts.dflutils import DflOptions, DflFiles
dfl_options = DflOptions(opts)

def list_models():
    return dfl_options.get_dfl_list()

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
        run(f'"{python}" -m pip install -U openmim', desc="Installing openmim", errdesc="Couldn't install openmim")
        run(f'"{python}" -m mim install mmcv-full', desc=f"Installing mmcv-full", errdesc=f"Couldn't install mmcv-full")
        run(f'"{python}" -m pip install mmdet', desc=f"Installing mmdet", errdesc=f"Couldn't install mmdet")

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

        if is_img2img:
            info = gr.HTML("<p style=\"\"></p>")
        else:
            info = gr.HTML("")
        with gr.Group():
            with gr.Row():
                dfm_model_dropdown = gr.Dropdown(label="DFM Model", choices=list_models(), value="None", visible=True, type="value")
                #dfm_model_dropdown_include_downloadable = gr.Checkbox(label="Include downloadable")
                create_refresh_button(dfm_model_dropdown, lambda: None, lambda: {"choices": list_models()}, "refresh_dfm_model_list")
            with gr.Row():
                info_upload = gr.HTML(
                    "<p style=\"padding-left: 5px;\">You can put the models into <b>" + str(dfl_options.dfl_path) + "</b> Change the folder in the Settings</p>")

            with gr.Row():
                image_return_original_checkbox = gr.Checkbox(label="Return original image")
                save_training_data_checkbox = gr.Checkbox(label="Save faces for training")
                only_training_images_checkbox = gr.Checkbox(label="Extract only training faces")
            with gr.Row():
                enable_detection_detailer_face_checkbox = gr.Checkbox(label="Enable Detection Detailer for face", value=True)
                save_detection_detailer_image_checkbox = gr.Checkbox(label="Save Detection Detailer Image")
                use_cpu_checkbox = gr.Checkbox(label="Use CPU", value=True)

        with gr.Tab("Face Detector"):
            with gr.Row():
                step_1_detector_input = gr.Dropdown(label="Detector", choices=list_detectors(), value="YoloV5", visible=True, type="value")
                step_1_window_size_input = gr.Dropdown(label="Window size", choices=["Auto", "512", "320", "160"], value="Auto", type="value")
                step_1_threshold_input = gr.Number(label="Threshold", min_value=0, max_value=100, step=1, value=50)
            with gr.Row():
                step_1_max_faces_input = gr.Number(label="Max Faces", min_value=1, max_value=None, value=3)
                step_1_sort_by_input = gr.Dropdown(label="Sort By", choices=["Largest", "Dist from center", "From left to right", "From top to bottom", "From bottom to top"], value="Largest")

        with gr.Tab("Face Marker"):
            with gr.Row():
                step_2_marker_input = gr.Dropdown(label="Marker", choices=list_markers(), value="InsightFace_2D106", visible=True, type="value")
                step_2_marker_coverage_input = gr.Number(label="Marker coverage", min_value=0, max_value=3, step=0.1, value=1.6)

        with gr.Tab("Face Aligner"):
            with gr.Row():
                step_3_align_mode_input = gr.Dropdown(label="Align mode", choices=list_align_modes(), value="From points", visible=True, type="value")
                step_3_face_coverage_input = gr.Number(label="Face coverage", min_value=0, max_value=3, step=0.1, value=2.2)
                step_3_resolution_input = gr.Number(label="Resolution", min_value=16, max_value=1024, step=16, value=320)
            with gr.Row():
                step_3_exclude_moving_parts_input = gr.Checkbox(label="Exclude moving parts", value=True)
                step_3_head_mode_input = gr.Checkbox(label="Head mode", value=False)
                step_3_freeze_z_rotation_input = gr.Checkbox(label="Freeze Z rotation", value=False)
            with gr.Row():
                step_3_x_offset_input = gr.Number(label="X offset", min_value=0, max_value=200, step=0.1, value=0)
                step_3_y_offset_input = gr.Number(label="Y offset", min_value=0, max_value=200, step=0.1, value=0)

        with gr.Tab("Face Swapper"):
            with gr.Row():
                step_4_swap_all_faces_input = gr.Checkbox(label="Swap all faces", value=True)
                step_4_face_id_input = gr.Number(label="Face ID", min_value=0, max_value=24, step=1, value=0)
                step_4_two_pass_input = gr.Checkbox(label="Two Pass", value=False)
                step_4_pre_sharpen_input = gr.Number(label="Pre-sharpen", min_value=0, max_value=100, step=1, value=0.5)
            with gr.Row():
                step_4_pre_gamma_red_input = gr.Number(label="Pre-gamma Red", min_value=1, max_value=100, step=1, value=1)
                step_4_pre_gamma_green_input = gr.Number(label="Pre-gamma Green", min_value=1, max_value=100, step=1, value=1)
                step_4_pre_gamma_blue_input = gr.Number(label="Pre-gamma Blue", min_value=1, max_value=100, step=1, value=1)
            with gr.Row():
                step_4_post_gamma_red_input = gr.Number(label="Post-gamma Red", min_value=1, max_value=100, step=1, value=1)
                step_4_post_gamma_green_input = gr.Number(label="Post-gamma Green", min_value=1, max_value=100, step=1, value=1)
                step_4_post_gamma_blue_input = gr.Number(label="Post-gamma Blue", min_value=1, max_value=100, step=1, value=1)


        with gr.Tab("Face Merger"):
            with gr.Row():
                step_5_x_offset_input = gr.Number(label="Face X offset", min_value=0, max_value=10, step=1, value=0)
                step_5_y_offset_input = gr.Number(label="Face Y offset", min_value=0, max_value=10, step=1, value=0)
                step_5_face_scale_input = gr.Number(label="Face scale", min_value=0, max_value=10, step=1, value=1)
            with gr.Row():
                step_5_face_mask_src_input = gr.Checkbox(label="Face mask (SRC)", value=True)
                step_5_face_celeb_input = gr.Checkbox(label="Face mask (CELEB)", value=True)
                step_5_face_lmrks_input = gr.Checkbox(label="Face mask (LMRKS)", value=False)
            with gr.Row():
                step_5_face_mask_erode_input = gr.Number(label="Face mask erode", min_value=0, max_value=100, step=1, value=5)
                step_5_face_mask_blur_input = gr.Number(label="Face mask blur", min_value=0, max_value=100, step=1, value=25)
                step_5_color_transfer_input = gr.Dropdown(label="Color transfer", choices=["none", "rct"], value="rct", visible=True, type="value")
            with gr.Row():
                step_5_color_compression_input = gr.Number(label="Color compression", min_value=0, max_value=100, step=1, value=0)
                step_5_face_opacity_input = gr.Number(label="Face opacity", min_value=0, max_value=1, step=1, value=1)
                #step_5_interpolation_input = gr.Dropdown(label="Interpolation", choices=["bilinear", "bicubic", "lanczos4"], value="bilinear", visible=True, type="value")
                step_5_interpolation_input = gr.Dropdown(label="Interpolation", choices=["bilinear"], value="bilinear", visible=True, type="value")

        with gr.Group():
            with gr.Row():
                info = gr.HTML("<p style=\"padding-left: 5px;\">Note: If the output results are not a good quality you can fine tune them "
                               "by using the <b>Save images for training</b> checkbox to collect input training images to further "
                               "train the DFM model you have in the DeepFaceLab tab.</p>")

        return [info,
                dfm_model_dropdown,
                image_return_original_checkbox,
                save_training_data_checkbox,
                only_training_images_checkbox,
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

    def process_frame(self, orig_image, dfm_model_dropdown,   step_1_detector_input,
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
        img_array = np.array(orig_image)
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

        step_1_threshold = step_1_threshold_input

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

        img = execute_deep_face_live(numpy_image=img_array,
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
                                     step_5_face_opacity=step_5_face_opacity_input
                                     )
        return img

    def run(self, p,
            info,
            dfm_model_dropdown,
            image_return_original_checkbox,
            save_training_data_checkbox,
            only_training_images_checkbox,
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
        initial_info = None
        seed = p.seed
        batch_size = p.batch_size
        itterations = p.n_iter
        #p.n_iter = 1
        p.do_not_save_grid = True
        p.do_not_save_samples = True
        is_txt2img = isinstance(p, StableDiffusionProcessingTxt2Img)
        print(step_1_threshold_input)
        if (not is_txt2img):
            orig_image = p.init_images[0]
        else:
            p_txt = p
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

        output_images = []
        processed = processing.process_images(p_txt)
        state.job_count = len(processed.images)
        print(f"Itterations: {itterations}")
        print(batch_size)
        for current_image in processed.images:
            devices.torch_gc()
            if is_txt2img:
                init_image = current_image
            else:
                init_image = orig_image

            if image_return_original_checkbox or only_training_images_checkbox or dfm_model_dropdown == "None":
                output_images.append(init_image)

            print(f"Generation {p} out of {state.job_count}")

            if enable_detection_detailer_face_checkbox:
                ddscript = DetectionDetailerScript()
                init_image = ddscript.run(p=p, init_image=init_image)
                if save_detection_detailer_image_checkbox:
                    output_images.append(init_image)

            # Primary run
            if dfm_model_dropdown != "None" and not only_training_images_checkbox:
                output_images.append(self.process_frame(orig_image=init_image,
                                                        dfm_model_dropdown=dfm_model_dropdown,
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
                                                        step_5_face_opacity_input=step_5_face_opacity_input
                                                        ))

        if (initial_info is None):
            initial_info = ""

        return Processed(p, output_images, seed, initial_info)



def get_device():
    device_id = shared.cmd_opts.device_id
    if device_id is not None:
        cuda_device = f"cuda:{device_id}"
    else:
        cuda_device = "cpu"
    return cuda_device