import os
import sys

# Get the absolute path of the directory containing the importing file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the relative path to the importing file to the system path
sys.path.append(os.path.join(current_dir, '../repo/dflive/'))

import numpy as np
from enum import IntEnum
from xlib import avecl as lib_cl
from xlib.image import ImageProcessor
from xlib.face import FRect, ELandmarks2D, FLandmarks2D, FPose
from xlib.python import all_is_not_None
from modelhub import onnx as onnx_models
from modelhub import cv as cv_models
from apps.DeepFaceLive.backend.BackendBase import (BackendFaceSwapInfo)
from pathlib import Path
from modelhub.DFLive.DFMModel import DFMModel

class DetectorType(IntEnum):
    CENTER_FACE = 0
    S3FD = 1
    YOLOV5 = 2

class FaceSortBy(IntEnum):
    LARGEST = 0
    DIST_FROM_CENTER = 1
    LEFT_RIGHT = 2
    RIGHT_LEFT = 3
    TOP_BOTTOM = 4
    BOTTOM_TOP = 5

def step_1_face_detector(frame_image,
                         device_info,
                         detector_type=DetectorType.YOLOV5,
                         threshold=0.5,
                         fixed_window_size=320,
                         sort_by=FaceSortBy.LARGEST,
                         max_faces=3):
    _, H, W, _ = ImageProcessor(frame_image).get_dims()

    rects = []
    if detector_type == DetectorType.CENTER_FACE:
        centerFace = onnx_models.CenterFace(device_info)
        rects = centerFace.extract(frame_image, threshold=threshold, fixed_window=fixed_window_size)[0]
    elif detector_type == DetectorType.S3FD:
        sd3d = onnx_models.S3FD(device_info)
        rects = sd3d.extract(frame_image, threshold=threshold, fixed_window=fixed_window_size)[0]
    elif detector_type == DetectorType.YOLOV5:
        yoloV5Face = onnx_models.YoloV5Face(device_info)
        rects = yoloV5Face.extract(frame_image, threshold=threshold, fixed_window=fixed_window_size)[0]

    # to list of FaceURect
    rects = [FRect.from_ltrb((l / W, t / H, r / W, b / H)) for l, t, r, b in rects]

    # sort
    if sort_by == FaceSortBy.LARGEST:
        rects = FRect.sort_by_area_size(rects)
    elif sort_by == FaceSortBy.DIST_FROM_CENTER:
        rects = FRect.sort_by_dist_from_2D_point(rects, 0.5, 0.5)
    elif sort_by == FaceSortBy.LEFT_RIGHT:
        rects = FRect.sort_by_dist_from_horizontal_point(rects, 0)
    elif sort_by == FaceSortBy.RIGHT_LEFT:
        rects = FRect.sort_by_dist_from_horizontal_point(rects, 1)
    elif sort_by == FaceSortBy.TOP_BOTTOM:
        rects = FRect.sort_by_dist_from_vertical_point(rects, 0)
    elif sort_by == FaceSortBy.BOTTOM_TOP:
        rects = FRect.sort_by_dist_from_vertical_point(rects, 1)

    return_faces = []
    if len(rects) != 0:
        if max_faces != 0 and len(rects) > max_faces:
            rects = rects[:max_faces]

        for face_id, face_urect in enumerate(rects):
            if face_urect.get_area() != 0:
                fsi = BackendFaceSwapInfo()
                fsi.image_name = ""
                fsi.face_urect = face_urect
                return_faces.append(fsi)

    return return_faces

class MarkerType(IntEnum):
    OPENCV_LBF = 0
    GOOGLE_FACEMESH = 1
    INSIGHT_2D106 = 2

def step_2_face_marker(frame_image,
                       faces,
                       device_info,
                       marker_type=MarkerType.INSIGHT_2D106,
                       marker_state_coverage=1.6):
    if marker_type == MarkerType.OPENCV_LBF:
        opencv_lbf = cv_models.FaceMarkerLBF()
    elif marker_type == MarkerType.GOOGLE_FACEMESH:
        google_facemesh = onnx_models.FaceMesh(device_info)
    elif marker_type == MarkerType.INSIGHT_2D106:
        insightface_2d106 = onnx_models.InsightFace2D106(device_info)

    marker_coverage = marker_state_coverage
    if marker_coverage is None:
        if marker_type == MarkerType.OPENCV_LBF:
            marker_coverage = 1.1
        elif marker_type == MarkerType.GOOGLE_FACEMESH:
            marker_coverage = 1.4
        elif marker_type == MarkerType.INSIGHT_2D106:
            marker_coverage = 1.6

    is_opencv_lbf = marker_type == MarkerType.OPENCV_LBF and opencv_lbf is not None
    is_google_facemesh = marker_type == MarkerType.GOOGLE_FACEMESH and google_facemesh is not None
    is_insightface_2d106 = marker_type == MarkerType.INSIGHT_2D106 and insightface_2d106 is not None

    for face_iterration in enumerate(faces):
        face_id = face_iterration[0]
        fsi = face_iterration[1]
        if fsi.face_urect is not None:
            # Cut the face to feed to the face marker
            face_image, face_uni_mat = fsi.face_urect.cut(frame_image, marker_coverage,
                                                          256 if is_opencv_lbf else \
                                                              192 if is_google_facemesh else \
                                                                  192 if is_insightface_2d106 else 0)
            _, H, W, _ = ImageProcessor(face_image).get_dims()
            if is_opencv_lbf:
                lmrks = opencv_lbf.extract(face_image)[0]
            elif is_google_facemesh:
                lmrks = google_facemesh.extract(face_image)[0]
            elif is_insightface_2d106:
                lmrks = insightface_2d106.extract(face_image)[0]

            if is_google_facemesh:
                fsi.face_pose = FPose.from_3D_468_landmarks(lmrks)

            if is_opencv_lbf:
                lmrks /= (W, H)
            elif is_google_facemesh:
                lmrks = lmrks[..., 0:2] / (W, H)
            elif is_insightface_2d106:
                lmrks = lmrks[..., 0:2] / (W, H)

            face_ulmrks = FLandmarks2D.create(ELandmarks2D.L68 if is_opencv_lbf else \
                                                  ELandmarks2D.L468 if is_google_facemesh else \
                                                      ELandmarks2D.L106 if is_insightface_2d106 else None,
                                              lmrks)
            face_ulmrks = face_ulmrks.transform(face_uni_mat, invert=True)
            fsi.face_ulmrks = face_ulmrks

    return faces

class AlignMode(IntEnum):
    FROM_RECT = 0
    FROM_POINTS = 1
    FROM_STATIC_RECT = 2

def step_3_face_aligner(frame_image,
                        faces,
                        align_mode=AlignMode.FROM_POINTS,
                        head_mode=False,
                        freeze_z_rotation=False,
                        resolution=320,
                        face_coverage=2.2,
                        x_offset=0,
                        y_offset=0,
                        exclude_moving_parts=False):
    for face_iterration in enumerate(faces):
        face_id = face_iterration[0]
        fsi = face_iterration[1]
        head_yaw = None
        if head_mode or freeze_z_rotation:
            if fsi.face_pose is not None:
                head_yaw = fsi.face_pose.as_radians()[1]

        face_ulmrks = fsi.face_ulmrks
        if face_ulmrks is not None:
            fsi.face_resolution = resolution

            H, W = frame_image.shape[:2]
            if align_mode == AlignMode.FROM_RECT:
                face_align_img, uni_mat = fsi.face_urect.cut(frame_image, coverage=face_coverage,
                                                             output_size=resolution,
                                                             x_offset=x_offset, y_offset=y_offset)

            elif align_mode == AlignMode.FROM_POINTS:
                face_align_img, uni_mat = face_ulmrks.cut(frame_image,
                                                          face_coverage + (1.0 if head_mode else 0.0),
                                                          resolution,
                                                          exclude_moving_parts=exclude_moving_parts,
                                                          head_yaw=head_yaw,
                                                          x_offset=x_offset,
                                                          y_offset=y_offset - 0.08 + (
                                                              -0.50 if head_mode else 0.0),
                                                          freeze_z_rotation=freeze_z_rotation)
            elif align_mode == AlignMode.FROM_STATIC_RECT:
                rect = FRect.from_ltrb([0.5 - (fsi.face_resolution / W) / 2, 0.5 - (fsi.face_resolution / H) / 2,
                                        0.5 + (fsi.face_resolution / W) / 2, 0.5 + (fsi.face_resolution / H) / 2, ])
                face_align_img, uni_mat = rect.cut(frame_image, coverage=face_coverage,
                                                   output_size=resolution,
                                                   x_offset=x_offset, y_offset=y_offset)

            fsi.face_align_image_name = f'{face_id}_aligned'
            fsi.image_to_align_uni_mat = uni_mat
            fsi.face_align_ulmrks = face_ulmrks.transform(uni_mat)

            # Due to FaceAligner is not well loaded, we can make lmrks mask here
            face_align_lmrks_mask_img = fsi.face_align_ulmrks.get_convexhull_mask(face_align_img.shape[:2],
                                                                                  color=(255,), dtype=np.uint8)
            fsi.face_align_lmrks_mask_name = f'{face_id}_aligned_lmrks_mask'
            fsi.aligned_image = face_align_img
            fsi.aligned_lmrks_mask_image = face_align_lmrks_mask_img

    return faces

def step_4_face_swapper(face_align_images,
                        dfm_model,
                        device_info,
                        swap_all_faces=True,
                        selected_face_id=1,
                        pre_gamma=[1,1,1],
                        post_gamma=[1,1,1],
                        presharpen_amount=2,
                        two_pass=False,
                        morph_factor=0):

    face_output = []

    for image_id, faces in enumerate(face_align_images):
        face_iterration_output = []
        for face_align_image in faces:
            pre_gamma_red, pre_gamma_green, pre_gamma_blue = list(pre_gamma)
            post_gamma_red, post_gamma_green, post_gamma_blue = list(post_gamma)

            fai_ip = ImageProcessor(face_align_image)
            if presharpen_amount != 0:
                fai_ip.gaussian_sharpen(sigma=1.0, power=presharpen_amount)

            if pre_gamma_red != 1.0 or pre_gamma_green != 1.0 or pre_gamma_blue != 1.0:
                fai_ip.gamma(pre_gamma_red, pre_gamma_green, pre_gamma_blue)
            face_align_image = fai_ip.get_image('NHWC')

            celeb_face, celeb_face_mask_img, face_align_mask_img = dfm_model.convert(face_align_image,
                                                                                     morph_factor=morph_factor)
            celeb_face, celeb_face_mask_img, face_align_mask_img = celeb_face[0], celeb_face_mask_img[0], \
            face_align_mask_img[0]

            if two_pass:
                celeb_face, celeb_face_mask_img, _ = dfm_model.convert(celeb_face,
                                                                       morph_factor=morph_factor)
                celeb_face, celeb_face_mask_img = celeb_face[0], celeb_face_mask_img[0]

            if post_gamma_red != 1.0 or post_gamma_blue != 1.0 or post_gamma_green != 1.0:
                celeb_face = ImageProcessor(celeb_face).gamma(post_gamma_red, post_gamma_blue,
                                                              post_gamma_green).get_image('HWC')

            face_iterration_output.append([face_align_mask_img, celeb_face, celeb_face_mask_img])
        face_output.append(face_iterration_output)

    return face_output

_cpu_interp = {'bilinear' : ImageProcessor.Interpolation.LINEAR,
               'bicubic'  : ImageProcessor.Interpolation.CUBIC,
               'lanczos4' : ImageProcessor.Interpolation.LANCZOS4}

def step_5_merge_on_cpu(frame_image, face_resolution, face_align_img, face_align_mask_img, face_align_lmrks_mask_img, face_swap_img, face_swap_mask_img, aligned_to_source_uni_mat, frame_width, frame_height, do_color_compression, interpolation, face_mask_source, face_mask_celeb, face_mask_lmrks, face_mask_erode, face_mask_blur, color_transfer, face_opacity, color_compression ):
    import numexpr as ne
    interpolation = _cpu_interp[interpolation]

    frame_image = ImageProcessor(frame_image).to_ufloat32().get_image('HWC')

    masks = []
    if face_mask_source:
        masks.append( ImageProcessor(face_align_mask_img).to_ufloat32().get_image('HW') )
    if face_mask_celeb:
        masks.append( ImageProcessor(face_swap_mask_img).to_ufloat32().get_image('HW') )
    if face_mask_lmrks:
        masks.append( ImageProcessor(face_align_lmrks_mask_img).to_ufloat32().get_image('HW') )

    masks_count = len(masks)
    if masks_count == 0:
        face_mask = np.ones(shape=(face_resolution, face_resolution), dtype=np.float32)
    else:
        face_mask = masks[0]
        for i in range(1, masks_count):
            face_mask *= masks[i]

    # Combine face mask
    face_mask = ImageProcessor(face_mask).erode_blur(face_mask_erode, face_mask_blur, fade_to_border=True).get_image('HWC')
    frame_face_mask = ImageProcessor(face_mask).warp_affine(aligned_to_source_uni_mat, frame_width, frame_height).clip2( (1.0/255.0), 0.0, 1.0, 1.0).get_image('HWC')

    face_swap_ip = ImageProcessor(face_swap_img).to_ufloat32()

    if color_transfer == 'rct':
        face_swap_img = face_swap_ip.rct(like=face_align_img, mask=face_mask, like_mask=face_mask)

    frame_face_swap_img = face_swap_ip.warp_affine(aligned_to_source_uni_mat, frame_width, frame_height, interpolation=interpolation).get_image('HWC')

    # Combine final frame
    opacity = np.float32(face_opacity)
    one_f = np.float32(1.0)
    if opacity == 1.0:
        out_merged_frame = ne.evaluate('frame_image*(one_f-frame_face_mask) + frame_face_swap_img*frame_face_mask')
    else:
        out_merged_frame = ne.evaluate('frame_image*(one_f-frame_face_mask) + frame_image*frame_face_mask*(one_f-opacity) + frame_face_swap_img*frame_face_mask*opacity')

    if do_color_compression and color_compression != 0:
        color_compression = max(4, (127.0 - color_compression) )
        out_merged_frame *= color_compression
        np.floor(out_merged_frame, out=out_merged_frame)
        out_merged_frame /= color_compression
        out_merged_frame += 2.0 / color_compression

    return out_merged_frame

_gpu_interp = {'bilinear' : lib_cl.EInterpolation.LINEAR,
                   'bicubic'  : lib_cl.EInterpolation.CUBIC,
                   'lanczos4' : lib_cl.EInterpolation.LANCZOS4}

_n_mask_multiply_op_text = [ f"float X = {'*'.join([f'(((float)I{i}) / 255.0)' for i in range(n)])}; O = (X <= 0.5 ? 0 : 1);" for n in range(5) ]

def step_5_merge_on_gpu(frame_image, face_resolution, face_align_img, face_align_mask_img, face_align_lmrks_mask_img, face_swap_img, face_swap_mask_img, aligned_to_source_uni_mat, frame_width, frame_height, do_color_compression, interpolation, face_mask_source, face_mask_celeb, face_mask_lmrks, face_mask_erode, face_mask_blur, color_transfer, face_opacity, color_compression ):
    interpolation = _gpu_interp[interpolation]

    masks = []
    if face_mask_source:
        masks.append( lib_cl.Tensor.from_value(face_align_mask_img) )
    if face_mask_celeb:
        masks.append( lib_cl.Tensor.from_value(face_swap_mask_img) )
    if face_mask_lmrks:
        masks.append( lib_cl.Tensor.from_value(face_align_lmrks_mask_img) )

    masks_count = len(masks)
    if masks_count == 0:
        face_mask_t = lib_cl.Tensor(shape=(face_resolution, face_resolution), dtype=np.float32, initializer=lib_cl.InitConst(1.0))
    else:
        face_mask_t = lib_cl.any_wise(_n_mask_multiply_op_text[masks_count], *masks, dtype=np.uint8).transpose( (2,0,1) )

    face_mask_t = lib_cl.binary_morph(face_mask_t, face_mask_erode, face_mask_blur, fade_to_border=True, dtype=np.float32)
    face_swap_img_t  = lib_cl.Tensor.from_value(face_swap_img ).transpose( (2,0,1), op_text='O = ((O_TYPE)I) / 255.0', dtype=np.float32)

    if color_transfer == 'rct':
        face_align_img_t = lib_cl.Tensor.from_value(face_align_img).transpose( (2,0,1), op_text='O = ((O_TYPE)I) / 255.0', dtype=np.float32)
        face_swap_img_t = lib_cl.rct(face_swap_img_t, face_align_img_t, target_mask_t=face_mask_t, source_mask_t=face_mask_t)

    frame_face_mask_t     = lib_cl.remap_np_affine(face_mask_t,     aligned_to_source_uni_mat, interpolation=lib_cl.EInterpolation.LINEAR, output_size=(frame_height, frame_width), post_op_text='O = (O <= (1.0/255.0) ? 0.0 : O > 1.0 ? 1.0 : O);' )
    frame_face_swap_img_t = lib_cl.remap_np_affine(face_swap_img_t, aligned_to_source_uni_mat, interpolation=interpolation, output_size=(frame_height, frame_width), post_op_text='O = clamp(O, 0.0, 1.0);' )

    frame_image_t = lib_cl.Tensor.from_value(frame_image).transpose( (2,0,1), op_text='O = ((float)I) / 255.0;' if frame_image.dtype == np.uint8 else None,
                                                                              dtype=np.float32 if frame_image.dtype == np.uint8 else None)

    opacity = face_opacity
    if opacity == 1.0:
        frame_final_t = lib_cl.any_wise('O = I0*(1.0-I1) + I2*I1', frame_image_t, frame_face_mask_t, frame_face_swap_img_t, dtype=np.float32)
    else:
        frame_final_t = lib_cl.any_wise('O = I0*(1.0-I1) + I0*I1*(1.0-I3) + I2*I1*I3', frame_image_t, frame_face_mask_t, frame_face_swap_img_t, np.float32(opacity), dtype=np.float32)

    if do_color_compression and color_compression != 0:
        color_compression = max(4, (127.0 - color_compression) )
        frame_final_t = lib_cl.any_wise('O = ( floor(I0 * I1) / I1 ) + (2.0 / I1);', frame_final_t, np.float32(color_compression))

    return frame_final_t.transpose( (1,2,0) ).np()

def step_5_face_merger(frame_image,
                       faces,
                       device_info,
                       face_x_offset=0,
                       face_y_offset=0,
                       face_scale=1,
                       interpolation='bilinear',
                       face_mask_source=True,
                       face_mask_celeb=True,
                       face_mask_lmrks=False,
                       face_mask_erode=5,
                       face_mask_blur=25,
                       color_transfer=False,
                       face_opacity=1,
                       color_compression=0):

    merged_frame = frame_image

    if merged_frame is not None:
        fsi_list = faces
        fsi_list_len = len(fsi_list)
        has_merged_faces = False

        for face_iterration in enumerate(faces):
            fsi_id = face_iterration[0]
            fsi = face_iterration[1]

            image_to_align_uni_mat = fsi.image_to_align_uni_mat
            face_resolution = fsi.face_resolution

            face_align_img = fsi.aligned_image
            face_align_lmrks_mask_img = fsi.aligned_lmrks_mask_image
            face_align_mask_img = fsi.face_align_mask_image
            face_swap_img = fsi.face_swap_image_image
            face_swap_mask_img = fsi.face_swap_mask_image

            if all_is_not_None(face_resolution, face_align_img, face_align_mask_img, face_swap_img,
                               face_swap_mask_img, image_to_align_uni_mat):
                face_height, face_width = face_align_img.shape[:2]
                frame_height, frame_width = merged_frame.shape[:2]
                aligned_to_source_uni_mat = image_to_align_uni_mat.invert()
                aligned_to_source_uni_mat = aligned_to_source_uni_mat.source_translated(-face_x_offset,
                                                                                        -face_y_offset)
                aligned_to_source_uni_mat = aligned_to_source_uni_mat.source_scaled_around_center(face_scale,
                                                                                                  face_scale)
                aligned_to_source_uni_mat = aligned_to_source_uni_mat.to_exact_mat(face_width, face_height,
                                                                                   frame_width, frame_height)

                do_color_compression = fsi_id == fsi_list_len - 1
                if str(device_info) == 'CPU':
                    merged_frame = step_5_merge_on_cpu(merged_frame, face_resolution, face_align_img,
                                                      face_align_mask_img, face_align_lmrks_mask_img, face_swap_img,
                                                      face_swap_mask_img, aligned_to_source_uni_mat, frame_width,
                                                      frame_height, do_color_compression, interpolation, face_mask_source, face_mask_celeb, face_mask_lmrks, face_mask_erode, face_mask_blur, color_transfer, face_opacity, color_compression)
                else:
                    merged_frame = step_5_merge_on_gpu(merged_frame, face_resolution, face_align_img,
                                                      face_align_mask_img, face_align_lmrks_mask_img, face_swap_img,
                                                      face_swap_mask_img, aligned_to_source_uni_mat, frame_width,
                                                      frame_height, do_color_compression, interpolation, face_mask_source, face_mask_celeb, face_mask_lmrks, face_mask_erode, face_mask_blur, color_transfer, face_opacity, color_compression)

    return merged_frame

from xlib.onnxruntime import (get_available_devices_info)

def execute_deep_face_live_multiple( numpy_images,
                            dfm_path,
                            device_id = 0,
                            step_1_detector = DetectorType.YOLOV5,
                            step_1_window_size = 320,
                            step_1_threshold = 0.5,
                            step_1_max_faces = 3,
                            step_1_sort_by = FaceSortBy.LARGEST,
                            step_2_marker = MarkerType.INSIGHT_2D106,
                            step_2_marker_coverage = 1.6,
                            step_3_align_mode = AlignMode.FROM_POINTS,
                            step_3_face_coverage = 2.2,
                            step_3_resolution = 320,
                            step_3_exclude_moving_parts = True,
                            step_3_head_mode = False,
                            step_3_freeze_z_rotation = False,
                            step_3_x_offset = 0,
                            step_3_y_offset = 0,
                            step_4_swap_all_faces = True,
                            step_4_face_id = 0,
                            step_4_two_pass = False,
                            step_4_pre_sharpen = 0.5,
                            step_4_pre_gamma_red = 1,
                            step_4_pre_gamma_green = 1,
                            step_4_pre_gamma_blue = 1,
                            step_4_post_gamma_red = 1,
                            step_4_post_gamma_green = 1,
                            step_4_post_gamma_blue = 1,
                            step_5_x_offset=0,
                            step_5_y_offset=0,
                            step_5_face_scale=1,
                            step_5_face_mask_src=True,
                            step_5_face_celeb=True,
                            step_5_face_lmrks=False,
                            step_5_face_mask_erode=5,
                            step_5_face_mask_blur=25,
                            step_5_color_transfer='rct',
                            step_5_interpolation='bilinear',
                            step_5_color_compression=0,
                            step_5_face_opacity=1
                           ):
    faces_array = []
    for image_id, numpy_image in enumerate(numpy_images):
        print("Numpy array shape:", numpy_image.shape)
        available_devices = get_available_devices_info()
        device_info = available_devices[int(device_id)]

        print("Step 1. Face Detector")
        faces = step_1_face_detector(frame_image=numpy_image,
                                     device_info=device_info,
                                     detector_type=int(step_1_detector),
                                     threshold=float(step_1_threshold),
                                     fixed_window_size=int(step_1_window_size),
                                     sort_by=int(step_1_sort_by),
                                     max_faces=int(step_1_max_faces))

        print("Step 2. Face Marker")
        faces = step_2_face_marker(frame_image=numpy_image,
                                   faces=faces,
                                   device_info=device_info,
                                   marker_type=step_2_marker,
                                   marker_state_coverage=step_2_marker_coverage)

        print("Step 3. Face Aligner")
        faces = step_3_face_aligner(frame_image=numpy_image,
                                    faces=faces,
                                    align_mode=step_3_align_mode,
                                    head_mode=step_3_head_mode,
                                    freeze_z_rotation=step_3_freeze_z_rotation,
                                    resolution=step_3_resolution,
                                    face_coverage=step_3_face_coverage,
                                    x_offset=step_3_x_offset,
                                    y_offset=step_3_y_offset,
                                    exclude_moving_parts=step_3_exclude_moving_parts)

        faces_array.append(faces)

    # Note: There is a very weird issue with Step 4, when run through web ui, faces come out bluish and the issue is
    # not present when run through the command line. This is a workaround to fix the issue. The issue itself is in the
    # tensorflow library, and it is not present in the onnx version of the library.

    print("Step 4. Face Swapper")
    faces_array = step_4_face_swapper_remote(faces_array=faces_array,
                                dfm_model=dfm_path,
                                device_info=device_info,
                                swap_all_faces=step_4_swap_all_faces,
                                selected_face_id=int(step_4_face_id),
                                pre_gamma=[step_4_pre_gamma_red, step_4_pre_gamma_green, step_4_pre_gamma_blue],
                                post_gamma=[step_4_post_gamma_red, step_4_post_gamma_green, step_4_post_gamma_blue],
                                presharpen_amount=step_4_pre_sharpen,
                                two_pass=step_4_two_pass)

    # print("Step 4. Face Swapper")
    # faces = step_4_face_swapper(faces=faces,
    #                             dfm_model=dfm_path,
    #                             device_info=device_info,
    #                             swap_all_faces=step_4_swap_all_faces,
    #                             selected_face_id=int(step_4_face_id),
    #                             pre_gamma=[step_4_pre_gamma_red, step_4_pre_gamma_green, step_4_pre_gamma_blue],
    #                             post_gamma=[step_4_post_gamma_red, step_4_post_gamma_green, step_4_post_gamma_blue],
    #                             presharpen_amount=step_4_pre_sharpen,
    #                             two_pass=step_4_two_pass)

    # for face_iterration in enumerate(faces):
    #     i = face_iterration[0]
    #     fsi = face_iterration[1]
    #     return fsi.face_swap_image_image
    #     return ImageProcessor(fsi.aligned_image).get_image('HWC')

    output_images = []
    for image_id, faces in enumerate(faces_array):
        print("Step 5. Face Merger")
        output_images.append(step_5_face_merger(frame_image=numpy_images[image_id],
                                  faces=faces,
                                  device_info=device_info,
                                  face_x_offset=int(step_5_x_offset),
                                  face_y_offset=int(step_5_y_offset),
                                  face_scale=int(step_5_face_scale),
                                  interpolation=str(step_5_interpolation),
                                  face_mask_source=bool(step_5_face_mask_src),
                                  face_mask_celeb=bool(step_5_face_celeb),
                                  face_mask_lmrks=bool(step_5_face_lmrks),
                                  face_mask_erode=int(step_5_face_mask_erode),
                                  face_mask_blur=int(step_5_face_mask_blur),
                                  color_transfer=str(step_5_color_transfer),
                                  face_opacity=int(step_5_face_opacity),
                                  color_compression=int(step_5_color_compression)))

    return output_images

def step_4_face_swapper_remote(faces_array,
                        dfm_model,
                        device_info,
                        swap_all_faces=True,
                        selected_face_id=1,
                        pre_gamma=[1,1,1],
                        post_gamma=[1,1,1],
                        presharpen_amount=5,
                        two_pass=False,
                        morph_factor=0):
    import tempfile
    import subprocess
    import os
    import json
    from PIL import Image
    import base64
    import cv2

    faces_tmp_images = []

    for image_id, faces in enumerate(faces_array):
        face_iterration_images = []
        for face_iterration in enumerate(faces):
            face_id = face_iterration[0]
            fsi = face_iterration[1]

            # Create a temporary file to write the PNG image data to
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                # Use the Pillow library to write the NumPy array to the temporary file as a PNG image
                img = Image.fromarray(fsi.aligned_image)
                img.save(tmp_file, "PNG")

                # Get the filename of the temporary file
                tmp_filename = tmp_file.name
                face_iterration_images.append(tmp_filename)
        faces_tmp_images.append(face_iterration_images)

    faces_tmp_images_json = base64.b64encode(json.dumps(faces_tmp_images).encode("utf-8")).decode("utf-8")
    current_file = os.path.abspath(__file__)

    python_call = ['python', current_file, faces_tmp_images_json, dfm_model]
    print("Calling: " + " ".join(str(x) for x in python_call))

    result = subprocess.run(python_call, capture_output=True, text=True)

    try:
        # Try to parse the JSON output
        result = json.loads(result.stdout)
    except json.JSONDecodeError:
        # Handle the case where the output is not valid JSON
        print("Output is not valid JSON")
        return

    for image_id, faces_output in enumerate(result):
        for face_id in enumerate(faces_output):
            print(face_id)
            fsi = faces_array[image_id][face_id[0]]
            swap_swap_image = cv2.imread(face_id[1][1])
            swap_swap_image = cv2.cvtColor(swap_swap_image, cv2.COLOR_BGR2RGB)
            fsi.face_swap_image_image = swap_swap_image
            fsi.face_align_mask_image = np.array(cv2.imread(face_id[1][0]))
            fsi.face_swap_mask_image = np.array(cv2.imread(face_id[1][2]))

    return faces_array

# Here is step 4 called through the remote function, which is called from the web UI
if __name__ == '__main__':
    import cv2
    import json
    import tempfile
    from PIL import Image
    import base64

    # Get the image file path and DFL path from command line arguments
    faces_tmp_images_json = sys.argv[1]
    dfm_model = sys.argv[2]
    device_id = 0
    available_devices = get_available_devices_info()
    device_info = available_devices[int(device_id)]
    try:
        # Encode the output as a JSON-encoded string
        decoded_string = base64.b64decode(faces_tmp_images_json).decode("utf-8")
        faces_tmp_images = json.loads(decoded_string)
        # Print the output to stdout
    except json.JSONDecodeError:
        error_dict = {"error": "faces_tmp_images_json is not readable. Invalid JSON."}
        error_json = json.dumps(error_dict)
        print(error_json)
        exit()

    faces_numpy = []
    for image_id, faces in enumerate(faces_tmp_images):
        face_iterration_images = []
        for face in faces:
            # Load the image using OpenCV
            image = cv2.imread(face)

            # Convert the image to a numpy array
            numpy_array = np.array(image)
            face_iterration_images.append(numpy_array)
        faces_numpy.append(face_iterration_images)

    path = Path(dfm_model)
    dfm_model = DFMModel(path, device_info)

    output = step_4_face_swapper(face_align_images=faces_numpy,
                                      dfm_model=dfm_model,
                                      device_info=device_info)
    output_files = []
    for image_id, output_data in enumerate(output):
        output_itteration = []
        for face in output_data:
            # Create a temporary file to write the PNG image data to
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                # Write the numpy array to an image file using OpenCV
                cv2.imwrite(tmp_file.name, face[0])

                # Get the filename of the temporary file
                face_align_mask_img = tmp_file.name

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file2:
                # Use the Pillow library to write the NumPy array to the temporary file as a PNG image
                cv2.imwrite(tmp_file2.name, face[1])

                # Get the filename of the temporary file
                celeb_face = tmp_file2.name

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file3:
                # Use the Pillow library to write the NumPy array to the temporary file as a PNG image
                cv2.imwrite(tmp_file3.name, face[2])

                # Get the filename of the temporary file
                celeb_face_mask_img = tmp_file3.name

            output_itteration.append([face_align_mask_img, celeb_face, celeb_face_mask_img])
        output_files.append(output_itteration)

    output_json = json.dumps(output_files)
    print(output_json)

# if __name__ == '__main__':
#     # Get the image file path and DFL path from command line arguments
#     image_path = sys.argv[1]
#     dfl_path = sys.argv[2]
#     device_id = sys.argv[3]
#
#     # Load the image using OpenCV
#     image = cv2.imread(image_path)
#
#     # Convert the image to a numpy array
#     numpy_array = np.array(image)
#
#     final_image = execute_deep_face_live(numpy_array, dfl_path, device_id)
#     img = ImageProcessor(final_image, copy=True).to_uint8().get_image('HWC')
#     cv2.imwrite('output.jpg', img)

