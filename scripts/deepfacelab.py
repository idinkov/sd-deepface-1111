import subprocess
import platform
import math
import json
import sys
import os
import re
from pathlib import Path
import shutil

import gradio as gr
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter
import cv2

from modules.ui import create_refresh_button
from modules.ui_common import folder_symbol
from modules.shared import opts, OptionInfo
from modules import shared, paths, script_callbacks

current_frame_set = []
current_frame_set_index = 0


def on_ui_tabs():
    dflFiles = DflFiles()
    dfl_options = DflOptions(opts)

    dfl_path = dfl_options.get_dfl_path()
    workspaces_path = dfl_options.get_workspaces_path()
    pak_path = dfl_options.get_pak_path()
    xseg_path = dfl_options.get_xseg_path()
    saehd_path = dfl_options.get_saehd_path()
    videos_path = dfl_options.get_videos_path()
    videos_frames_path = dfl_options.get_videos_frames_path()
    tmp_path = dfl_options.get_tmp_path()

    def get_dfl_list():
        return dflFiles.get_files_from_dir(dfl_path, [".dfl"])

    def get_pak_list():
        return dflFiles.get_files_from_dir(pak_path, [".pak"])

    def get_videos_list():
        return dflFiles.get_files_from_dir(videos_path, [".mp4", ".mkv"])

    def get_videos_list_full_path():
        return list(videos_path + "/" + v for v in get_videos_list())

    def get_workspaces_list():
        return dflFiles.get_folder_names_in_dir(workspaces_path)

    def get_saehd_models_list():
        return dflFiles.get_folder_names_in_dir(saehd_path)

    def get_xseg_models_list():
        return dflFiles.get_folder_names_in_dir(xseg_path)

    def render_train_saehd(gr):
        with gr.Row():
            model_saehd_dropdown = gr.Dropdown(choices=get_saehd_models_list(), elem_id="saehd_model_dropdown", label="SAEHD Model:", interactive=True)
            create_refresh_button(model_saehd_dropdown, lambda: None, lambda: {"choices": get_saehd_models_list()}, "refresh_saehd_model_list")
            model_saehd_create_new_model = gr.Checkbox(label="Create new model")
        with gr.Row():
            model_xseg_dropdown = gr.Dropdown(choices=get_xseg_models_list(), elem_id="xseg_model_dropdown", label="XSEG Model:", interactive=True)
            create_refresh_button(model_xseg_dropdown, lambda: None, lambda: {"choices": get_xseg_models_list()}, "refresh_xseg_model_list")

        with gr.Row():
            src_pak_dropdown = gr.Dropdown(choices=get_pak_list(), elem_id="src_pak_dropdown", label="Src Faceset:", interactive=True)
            create_refresh_button(src_pak_dropdown, lambda: None, lambda: {"choices": get_pak_list()}, "refresh_src_pak_dropdown")

        with gr.Row():
            dst_pak_dropdown = gr.Dropdown(choices=get_pak_list(), elem_id="dst_pak_dropdown", label="Dst Faceset:", interactive=True)
            create_refresh_button(dst_pak_dropdown, lambda: None, lambda: {"choices": get_pak_list()}, "refresh_dst_pak_dropdown")

        train_saehd = gr.Button(value="Train SAEHD", variant="primary")
        log_output = gr.HTML(value="")

    def render_faceset_extract(gr):
        with gr.Row():
            videos_dropdown = gr.Dropdown(choices=get_videos_list(), elem_id="videos_dropdown", label="Videos",
                                          interactive=True)
            create_refresh_button(videos_dropdown, lambda: None, lambda: {"choices": get_videos_list()},
                                  "refresh_videos_dropdown")
        faceset_output_facetype_dropdown = gr.Dropdown(choices=['half_face', 'full_face', 'whole_face', 'head', 'mark_only'], value="whole_face", label="Output face type", interactive=True)
        faceset_output_resolution_dropdown = gr.Dropdown(choices=["256x256","512x512","768x768","1024x1024"], value="512x512", label="Output resolution", interactive=True)
        faceset_output_type_dropdown = gr.Dropdown(choices=["jpg","png"], value="jpg", label="Output filetype",interactive=True)
        faceset_output_quality_dropdown = gr.Dropdown(choices=[90,100], value=100, label="Output quality",interactive=True)
        faceset_output_debug_dropdown = gr.Checkbox(value=False, label="Generate debug frames")
        faceset_extract_frames_button = gr.Button(value="Extract Frames Only", variant="primary")
        faceset_extract_button = gr.Button(value="Faceset Extract", variant="primary")
        faceset_extract_output = gr.Markdown()
        faceset_extract_button.click(DflAction.extract_frames, [videos_dropdown], faceset_extract_output)
        faceset_extract_button.click(DflAction.extract_faceset, [videos_dropdown,
                                                                faceset_output_facetype_dropdown,
                                                                faceset_output_resolution_dropdown,
                                                                faceset_output_type_dropdown,
                                                                faceset_output_quality_dropdown,
                                                                faceset_output_debug_dropdown], faceset_extract_output)
        faceset_extract_frames_button.click(DflAction.extract_frames,videos_dropdown, faceset_extract_output)



    def render_create_dfl(gr):
        train_saehd = gr.Button(value="Create DFL", variant="primary")

    def click_create_workspace(text):
        if text == "":
            return f"Error!"
        return f"Workspace " + text + " created"

    def render_classic_tabs():
        with gr.Row():
            workspace_dropdown = gr.Dropdown(choices=get_workspaces_list(), elem_id="workspace_dropdown", label="Current workspace:", interactive=True)
            create_refresh_button(workspace_dropdown, lambda: None, lambda: {"choices": get_workspaces_list()}, "refresh_workspace_list")
        with gr.Tab("Faceset Extract"):
            render_faceset_extract(gr)
        with gr.Tab("Train"):
            render_train_saehd(gr)
        with gr.Tab("Create DFL"):
            render_create_dfl(gr)

    def upload_files_videos(files):
        file_paths = [file.name for file in files]
        for file in files:
            DflFiles.copy_file_to_dest_dir(file.name, "video.mp4", videos_path)

        return file_paths

    def get_current_video_path(videos_dropdown):
        return gr.Textbox.update(value=get_current_video_path_only(videos_dropdown), visible=True)

    def get_current_video_path_only(videos_dropdown):
        return str(videos_path) + "/" + str(videos_dropdown)

    def action_delete_video(videos_dropdown):
        filePath = get_current_video_path_only(videos_dropdown)
        DflFiles.delete_file(str(filePath))
        return gr.Textbox.update(visible=False)

    def reset_video_dropdown(videos_dropdown):
        return gr.Dropdown.update(value="")

    def render_dataset_videos():
        upload_button = gr.UploadButton("Click to Upload Video/s", file_types=["video"], file_count="multiple")
        file_output = gr.File(label="Uploaded Video/s")
        upload_button.upload(upload_files_videos, upload_button, file_output, scroll_to_output=True)

    def render_dataset_xseg():
        upload_button_xseg = gr.UploadButton("Click to Upload XSEG model/s", file_types=["video"], file_count="multiple")
        file_output_xseg = gr.File(label="Uploaded XSEG model/s")
        upload_button_xseg.upload(upload_files_videos, upload_button_xseg, file_output_xseg, scroll_to_output=True)

    def render_dataset_saehd():
        upload_button_saehd = gr.UploadButton("Click to Upload SAEHD model/s", file_types=["video"], file_count="multiple")
        file_output_saehd = gr.File(label="Uploaded SAEHD model/s")
        upload_button_saehd.upload(upload_files_videos, upload_button_saehd, file_output_saehd, scroll_to_output=True)

    def render_dataset_facesets():
        upload_button_datasets = gr.UploadButton("Click to Upload Faceset/s", file_types=["video"], file_count="multiple")
        file_output_datasets = gr.File(label="Uploaded Faceset/s")
        upload_button_datasets.upload(upload_files_videos, upload_button_datasets, file_output_datasets, scroll_to_output=True)

    def render_dataset_dfl():
        upload_button_dfl = gr.UploadButton("Click to Upload DFL/s", file_types=["video"], file_count="multiple")
        file_output_dfl = gr.File(label="Uploaded DFL/s")
        upload_button_dfl.upload(upload_files_videos, upload_button_dfl, file_output_dfl, scroll_to_output=True)

    # Display contents in main tab "DeepFaceLab" in SD1111 UI
    with gr.Blocks(analytics_enabled=False) as training_picker:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("DeepFaceLab")
                render_classic_tabs()
                gr.Markdown("Creator")
                with gr.Tab("Workspace"):
                    text = gr.Textbox(value="", label="Name")
                    button = gr.Button(value="Create Workspace", variant="primary")
                    output1 = gr.Textbox(label="Status")
                    button.click(click_create_workspace, [text], output1)
                with gr.Tab("SAEHD Model"):
                    with gr.Tab("New Model"):
                        nothing = 0
                    with gr.Tab("Clone Existing Model"):
                        with gr.Row():
                            model_saehd_dropdown = gr.Dropdown(choices=get_saehd_models_list(), elem_id="saehd_model_dropdown", label="SAEHD Model:", interactive=True)
                            create_refresh_button(model_saehd_dropdown, lambda: None, lambda: {"choices": get_saehd_models_list()}, "refresh_saehd_model_list")

                    text = gr.Textbox(value="", label="Name")
                    button = gr.Button(value="Create SAEHD Model", variant="primary")
                    output1 = gr.Textbox(label="Status")
                    button.click(click_create_workspace, [text], output1)

                gr.Markdown("Fast tools")
                with gr.Tab("DFL Creator"):
                    nothing = 0

            with gr.Column(scale=2):
                gr.Markdown("Preview/Browser")
                with gr.Tabs() as tabs_preview:
                    with gr.TabItem("Status", id=0):
                        nothing = 0
                    with gr.TabItem("Workspaces", id=1):
                        nothing = 0
                    with gr.TabItem("Videos", id=2):
                        with gr.Tabs() as tabs_preview_videos:
                            with gr.TabItem("Browser", id=0):
                                browser_videos_gallery = gr.Gallery(fn=get_videos_list_full_path)
                                browser_videos_gallery.style(grid=4, height=4, container=True)
                            with gr.TabItem("Preview", id=1):
                                with gr.Row():
                                    videos_dropdown = gr.Dropdown(choices=get_videos_list(), elem_id="videos_dropdown", label="Videos:", interactive=True)
                                    create_refresh_button(videos_dropdown, lambda: None, lambda: {"choices": get_videos_list()}, "refresh_videos_dropdown")
                                with gr.Row():
                                    download_video = gr.Button(value="Download Video", variant="gray")
                                    delete_video = gr.Button(value="Delete Video", variant="red")
                                main_video_preview = gr.Video(interactive=None)
                                download_video.click(reset_video_dropdown, videos_dropdown, videos_dropdown)
                                delete_video.click(action_delete_video, videos_dropdown, main_video_preview)
                                videos_dropdown.change(get_current_video_path, videos_dropdown, main_video_preview)
                    with gr.TabItem("XSEG", id=3):
                        with gr.Tabs() as tabs_preview_xseg:
                            with gr.TabItem("Browser", id=0):
                                nothing = 0
                            with gr.TabItem("Viewer", id=1):
                                with gr.Row():
                                    xseg_dropdown = gr.Dropdown(choices=get_xseg_models_list(), elem_id="xseg_dropdown", label="XSEG Model:", interactive=True)
                                    create_refresh_button(xseg_dropdown, lambda: None, lambda: {"choices": get_xseg_models_list()}, "refresh_xseg_list")

                    with gr.TabItem("SAEHD", id=4):
                        with gr.Tabs() as tabs_preview_saehd:
                            with gr.TabItem("Browser", id=0):
                                nothing = 0
                            with gr.TabItem("Viewer", id=1):
                                with gr.Row():
                                    saehd_dropdown = gr.Dropdown(choices=get_saehd_models_list(), elem_id="saehd_dropdown", label="SAEHD Model:", interactive=True)
                                    create_refresh_button(saehd_dropdown, lambda: None, lambda: {"choices": get_saehd_models_list()}, "refresh_dfl_list")

                    with gr.TabItem("Facesets", id=5):
                        with gr.Tabs() as tabs_preview_facesets:
                            with gr.TabItem("Browser", id=0):
                                nothing = 0
                            with gr.TabItem("Viewer", id=1):
                                with gr.Row():
                                    faceset_dropdown = gr.Dropdown(choices=get_pak_list(), elem_id="faceset_dropdown", label="Faceset:", interactive=True)
                                    create_refresh_button(faceset_dropdown, lambda: None, lambda: {"choices": get_pak_list()}, "refresh_faceset_list")
                    with gr.TabItem("DFL", id=6):
                        with gr.Tabs() as tabs_preview_dfl:
                            with gr.TabItem("Browser", id=0):
                                nothing = 0
                            with gr.TabItem("Viewer", id=1):
                                with gr.Row():
                                    dfl_dropdown = gr.Dropdown(choices=get_dfl_list(), elem_id="dfl_dropdown", label="DFL:", interactive=True)
                                    create_refresh_button(dfl_dropdown, lambda: None, lambda: {"choices": get_dfl_list()}, "refresh_dfl_list")
            with gr.Column(scale=1):
                gr.Markdown("Upload")
                with gr.Tab("Videos"):
                    render_dataset_videos()
                with gr.Tab("XSEG"):
                    render_dataset_xseg()
                with gr.Tab("SAEHD"):
                    render_dataset_saehd()
                with gr.Tab("Facesets"):
                    render_dataset_facesets()
                with gr.Tab("DFL"):
                    render_dataset_dfl()

    return (training_picker, "DeepFaceLab", "deepfacelab"),


def on_ui_settings():
    dfl_path = Path(paths.script_path) / "deepfacelab"
    section = ('deepfacelab', "DeepFaceLab")
    opts.add_option("deepfacelab_dflab_repo_path", OptionInfo(str(dfl_path / "dflab-repo"), "Path to DeepFaceLab repo are located", section=section))
    opts.add_option("deepfacelab_dflive_repo_path", OptionInfo(str(dfl_path / "dflive-repo"), "Path to DeepFaceLive repo are located", section=section))
    opts.add_option("deepfacelab_workspaces_path", OptionInfo(str(dfl_path / "workspaces"), "Path to dir where DeepFaceLab workspaces are located", section=section))
    opts.add_option("deepfacelab_dfl_path", OptionInfo(str(dfl_path / "dfl-files"), "Path to read/write .dfl files from", section=section))
    opts.add_option("deepfacelab_pak_path", OptionInfo(str(dfl_path / "pak-files"), "Default facesets .pak image directory", section=section))
    opts.add_option("deepfacelab_xseg_path", OptionInfo(str(dfl_path / "xseg-models"), "Default XSeg path for XSeg models directory", section=section))
    opts.add_option("deepfacelab_saehd_path", OptionInfo(str(dfl_path / "saehd-models"), "Default path for SAEHD models directory", section=section))
    opts.add_option("deepfacelab_videos_path", OptionInfo(str(dfl_path / "videos"), "Default path for Videos for deepfacelab", section=section))
    opts.add_option("deepfacelab_videos_frames_path", OptionInfo(str(dfl_path / "videos-frames"), "Default path for Video Frames for deepfacelab", section=section))
    opts.add_option("deepfacelab_tmp_path", OptionInfo(str(dfl_path / "tmp"), "Default path for tmp actions for deepfacelab", section=section))


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)



import os
from pathlib import Path

class DflCommunicator:
    @staticmethod
    def extract_frames(video_path, frames_dir, file_format_output='jpg', file_format_output_quality=100):
        # Create the frames directory if it doesn't exist
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Get the frames per second (FPS) of the video
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Initialize a counter for the frames
        frame_count = 0

        # Loop through the frames of the video
        while cap.isOpened():
            # Read a frame from the video
            ret, frame = cap.read()

            # If there are no more frames, break out of the loop
            if not ret:
                break

            # Save the frame as a file in the frames directory
            file_extension = '.' + file_format_output
            frame_file = os.path.join(frames_dir, f"{frame_count:06d}{file_extension}")
            if file_format_output == 'jpg':
                cv2.imwrite(frame_file, frame, [int(cv2.IMWRITE_JPEG_QUALITY), file_format_output_quality])
            elif file_format_output == 'png':
                cv2.imwrite(frame_file, frame, [int(cv2.IMWRITE_PNG_COMPRESSION), file_format_output_quality])

            # Increment the frame counter
            frame_count += 1

        # Release the video capture object
        cap.release()

        # Return the number of frames extracted
        return frame_count

class DflFiles:

    @staticmethod
    def folder_exists(dir_path):
        """
        Check whether a directory exists.

        Returns True if the directory exists, False otherwise.
        """
        return os.path.exists(dir_path) and os.path.isdir(dir_path)

    @staticmethod
    def create_folder(path):
        """Create a folder at the specified path"""
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def delete_folder(path):
        """Delete a folder and all its contents at the specified path"""
        os.removedirs(path)

    @staticmethod
    def empty_folder(path):
        """Clear the contents of a folder at the specified path"""
        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

    @staticmethod
    def create_empty_file(path):
        """Create an empty file at the specified path"""
        open(path, 'a').close()

    @staticmethod
    def delete_file(path):
        """Delete a file at the specified path"""
        os.remove(path)

    @staticmethod
    def move_file(src, dst):
        """Move a file from the source path to the destination path"""
        os.replace(src, dst)

    @staticmethod
    def get_files_from_dir(base_dir, extension_list):
        """Return a list of file names in a directory with a matching file extension"""
        files = []
        for v in Path(base_dir).iterdir():
            if v.suffix in extension_list and not v.name.startswith('.ipynb'):
                files.append(v.name)
        return files

    @staticmethod
    def get_folder_names_in_dir(base_dir):
        """Return a list of folder names in a directory"""
        folders = []
        for v in Path(base_dir).iterdir():
            if v.is_dir() and not v.name.startswith('.ipynb'):
                folders.append(v.name)
        return folders

    @staticmethod
    def extract_archive(archive_path, dest_dir, dir_name):
        """
        Extract an archive file to a directory with a specified name.

        The extracted directory will be created inside the destination directory with the specified name.
        If the name is taken, a suffix will be added to create a unique name.

        Returns the full path of the directory where the contents were extracted.
        """
        suffix = ''
        extracted_dir_name = dir_name
        while os.path.exists(os.path.join(dest_dir, extracted_dir_name + suffix)):
            if not suffix:
                suffix = 1
            else:
                suffix += 1
            extracted_dir_name = f'{dir_name}_{suffix}'

        extracted_dir_path = os.path.join(dest_dir, extracted_dir_name)
        os.makedirs(extracted_dir_path, exist_ok=True)

        # Define a dictionary that maps file extensions to archive types
        archive_types = {
            '.zip': 'zip',
            '.rar': 'rar',
            '.7z': '7z',
            '.tar': 'tar',
            '.tar.gz': 'gztar',
            '.tgz': 'gztar'
        }

        # Determine the type of archive file based on the file extension
        file_ext = os.path.splitext(archive_path)[1].lower()

        # Use the appropriate function from the `shutil` library to extract the archive
        shutil.unpack_archive(archive_path, extracted_dir_path, archive_types[file_ext])

        return extracted_dir_path

    @staticmethod
    def copy_file_to_dest_dir(temp_file_path, file_original_name, dest_dir):
        """
        Copy a file to a destination directory using the original file name.

        If a file with the same name already exists in the destination directory, a suffix
        will be added to the file name until a unique name is found.

        Returns the full path of the copied file.
        """
        suffix = ''
        dest_file_name = file_original_name
        while os.path.exists(os.path.join(dest_dir, dest_file_name)):
            if not suffix:
                suffix = 1
            else:
                suffix += 1
            dest_file_name = f'{os.path.splitext(file_original_name)[0]}_{suffix}{os.path.splitext(file_original_name)[1]}'

        dest_file_path = os.path.join(dest_dir, dest_file_name)
        shutil.copyfile(temp_file_path, dest_file_path)

        return dest_file_path

class DflOptions:
    def __init__(self, opts):
        self.dfl_files = DflFiles()
        self.dfl_path = Path(opts.deepfacelab_dfl_path)
        self.workspaces_path = Path(opts.deepfacelab_workspaces_path)
        self.pak_path = Path(opts.deepfacelab_pak_path)
        self.xseg_path = Path(opts.deepfacelab_xseg_path)
        self.saehd_path = Path(opts.deepfacelab_saehd_path)
        self.videos_path = Path(opts.deepfacelab_videos_path)
        self.videos_frames_path = Path(opts.deepfacelab_videos_frames_path)
        self.tmp_path = Path(opts.deepfacelab_tmp_path)
        self.dflab_repo = Path(opts.deepfacelab_dflab_repo_path)
        self.dflive_repo = Path(opts.deepfacelab_dflive_repo_path)

        # Create dirs if not existing
        for p in [self.dfl_path, self.workspaces_path, self.pak_path, self.xseg_path, self.saehd_path, self.videos_path, self.videos_frames_path, self.tmp_path]:
            self.dfl_files.create_folder(p)

    def get_dfl_path(self):
        return self.dfl_path

    def get_workspaces_path(self):
        return self.workspaces_path

    def get_pak_path(self):
        return self.pak_path

    def get_xseg_path(self):
        return self.xseg_path

    def get_saehd_path(self):
        return self.saehd_path

    def get_videos_path(self):
        return self.videos_path

    def get_videos_frames_path(self):
        return self.videos_frames_path

    def get_tmp_path(self):
        return self.tmp_path

    def get_dflab_repo_path(self):
        return self.dflab_repo

    def get_dflive_repo_path(self):
        return self.dflive_repo

class DflAction:
    dflFiles = DflFiles()

    @staticmethod
    def extract_frames(videos_dropdown):
        return "Video: " + str(videos_dropdown)

    @staticmethod
    def extract_faceset(videos_dropdown,
                       faceset_output_facetype_dropdown,
                       faceset_output_resolution_dropdown,
                       faceset_output_type_dropdown,
                       faceset_output_quality_dropdown,
                       faceset_output_debug_dropdown):
        return "Video: " + str(videos_dropdown) + " Output Face Type: " + faceset_output_facetype_dropdown