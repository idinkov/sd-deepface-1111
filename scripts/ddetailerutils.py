import os
import sys
import cv2
from PIL import Image
import numpy as np
import gradio as gr

from modules import processing, images
from modules import scripts, script_callbacks, shared, devices, modelloader
from modules.processing import Processed, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img
from modules.shared import opts, cmd_opts, state
from modules.sd_models import model_hash
from modules.paths import models_path
from basicsr.utils.download_util import load_file_from_url

dd_models_path = os.path.join(models_path, "mmdet")

def list_models(model_path):
    model_list = modelloader.load_models(model_path=model_path, ext_filter=[".pth"])

    def modeltitle(path, shorthash):
        abspath = os.path.abspath(path)

        if abspath.startswith(model_path):
            name = abspath.replace(model_path, '')
        else:
            name = os.path.basename(path)

        if name.startswith("\\") or name.startswith("/"):
            name = name[1:]

        shortname = os.path.splitext(name.replace("/", "_").replace("\\", "_"))[0]

        return f'{name} [{shorthash}]', shortname

    models = []
    for filename in model_list:
        h = model_hash(filename)
        title, short_model_name = modeltitle(filename, h)
        models.append(title)

    return models

class DetectionDetailerScript():
    def run(self,
            p,
            model,
            model_name,
            init_image,
            dd_conf_a = 30,
            dd_dilation_factor_a = 4,
            dd_offset_x_a = 0,
            dd_offset_y_a = 0):

        new_image = p
        results_a = inference(init_image, model, model_name, dd_conf_a / 100.0)
        masks_a = create_segmasks(results_a)
        masks_a = dilate_masks(masks_a, dd_dilation_factor_a, 1)
        masks_a = offset_masks(masks_a, dd_offset_x_a, dd_offset_y_a)
        output_image = init_image
        gen_count = len(masks_a)
        if (gen_count > 0):
            #state.job_count += gen_count
            new_image.init_images = [init_image]
            new_image.batch_size = 1
            new_image.n_iter = 1
            for i in range(gen_count):
                new_image.image_mask = masks_a[i]
                processed = processing.process_images(p)
                new_image.seed = processed.seed + 1
                new_image.init_images = processed.images

            if (gen_count > 0):
                output_image = processed.images[0]

        return output_image

import mmcv
from mmdet.apis import (inference_detector,
                        init_detector)


def modeldataset(model_shortname):
    path = modelpath(model_shortname)
    if ("mmdet" in path and "segm" in path):
        dataset = 'coco'
    else:
        dataset = 'bbox'
    return dataset

def modelpath(model_shortname):
    return dd_models_path + "/" + model_shortname

def is_allblack(mask):
    cv2_mask = np.array(mask)
    return cv2.countNonZero(cv2_mask) == 0

def bitwise_and_masks(mask1, mask2):
    cv2_mask1 = np.array(mask1)
    cv2_mask2 = np.array(mask2)
    cv2_mask = cv2.bitwise_and(cv2_mask1, cv2_mask2)
    mask = Image.fromarray(cv2_mask)
    return mask

def subtract_masks(mask1, mask2):
    cv2_mask1 = np.array(mask1)
    cv2_mask2 = np.array(mask2)
    cv2_mask = cv2.subtract(cv2_mask1, cv2_mask2)
    mask = Image.fromarray(cv2_mask)
    return mask

def dilate_masks(masks, dilation_factor, iter=1):
    if dilation_factor == 0:
        return masks
    dilated_masks = []
    kernel = np.ones((dilation_factor, dilation_factor), np.uint8)
    for i in range(len(masks)):
        cv2_mask = np.array(masks[i])
        dilated_mask = cv2.dilate(cv2_mask, kernel, iter)
        dilated_masks.append(Image.fromarray(dilated_mask))
    return dilated_masks

def offset_masks(masks, offset_x, offset_y):
    if (offset_x == 0 and offset_y == 0):
        return masks
    offset_masks = []
    for i in range(len(masks)):
        cv2_mask = np.array(masks[i])
        offset_mask = cv2_mask.copy()
        offset_mask = np.roll(offset_mask, -offset_y, axis=0)
        offset_mask = np.roll(offset_mask, offset_x, axis=1)

        offset_masks.append(Image.fromarray(offset_mask))
    return offset_masks

def combine_masks(masks):
    initial_cv2_mask = np.array(masks[0])
    combined_cv2_mask = initial_cv2_mask
    for i in range(1, len(masks)):
        cv2_mask = np.array(masks[i])
        combined_cv2_mask = cv2.bitwise_or(combined_cv2_mask, cv2_mask)

    combined_mask = Image.fromarray(combined_cv2_mask)
    return combined_mask

def create_segmasks(results):
    segms = results[1]
    segmasks = []
    for i in range(len(segms)):
        cv2_mask = segms[i].astype(np.uint8) * 255
        mask = Image.fromarray(cv2_mask)
        segmasks.append(mask)

    return segmasks

def get_device():
    device_id = shared.cmd_opts.device_id
    if device_id is not None:
        cuda_device = f"cuda:{device_id}"
    else:
        cuda_device = "cpu"
    return cuda_device

def inference(image, model, modelname, conf_thres):
    path = modelpath(modelname)
    if ("mmdet" in path and "bbox" in path):
        results = inference_mmdet_bbox(image, model, modelname, conf_thres)
    elif ("mmdet" in path and "segm" in path):
        results = inference_mmdet_segm(image, model, modelname, conf_thres)
    return results

def preload_ddetailer_model(modelname):
    model_checkpoint = modelpath(modelname)
    model_config = os.path.splitext(model_checkpoint)[0] + ".py"
    model_device = get_device()
    return init_detector(model_config, model_checkpoint, device=model_device)

def inference_mmdet_segm(image, model, modelname, conf_thres):
    mmdet_results = inference_detector(model, np.array(image))
    bbox_results, segm_results = mmdet_results
    dataset = modeldataset(modelname)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_results)
    ]
    n, m = bbox_results[0].shape
    if (n == 0):
        return [[], []]
    labels = np.concatenate(labels)
    bboxes = np.vstack(bbox_results)
    segms = mmcv.concat_list(segm_results)
    filter_inds = np.where(bboxes[:, -1] > conf_thres)[0]
    results = [[], []]
    for i in filter_inds:
        results[0].append(bboxes[i])
        results[1].append(segms[i])

    return results

def inference_mmdet_bbox(image, model, modelname, conf_thres):
    results = inference_detector(model, np.array(image))
    cv2_image = np.array(image)
    cv2_image = cv2_image[:, :, ::-1].copy()
    cv2_gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

    segms = []
    for (x0, y0, x1, y1, conf) in results[0]:
        cv2_mask = np.zeros((cv2_gray.shape), np.uint8)
        cv2.rectangle(cv2_mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
        cv2_mask_bool = cv2_mask.astype(bool)
        segms.append(cv2_mask_bool)

    n, m = results[0].shape
    if (n == 0):
        return [[], []]
    bboxes = np.vstack(results[0])
    filter_inds = np.where(bboxes[:, -1] > conf_thres)[0]
    results = [[], []]
    for i in filter_inds:
        results[0].append(bboxes[i])
        results[1].append(segms[i])

    return results