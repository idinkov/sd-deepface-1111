import os
import shutil
from pathlib import Path
from modules import shared, paths, script_callbacks

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
    def get_files_from_dir(base_dir, extension_list, two_dimensions=False):
        """Return a list of file names in a directory with a matching file extension"""
        files = []
        for v in Path(base_dir).iterdir():
            if v.suffix in extension_list and not v.name.startswith('.ipynb'):
                if two_dimensions:
                    files.append([v.name, v.name])
                else:
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
        dfl_path = Path(paths.script_path) / "deepfacelab"
        scripts_path = Path(os.path.dirname(os.path.abspath(__file__))) / "../"
        print("Scripts path:" + str(scripts_path))

        self.dfl_path = Path(str(dfl_path / "dfl-files"))
        self.workspaces_path = Path(str(dfl_path / "workspaces"))
        self.pak_path = Path(str(dfl_path / "pak-files"))
        self.xseg_path = Path(str(dfl_path / "xseg-models"))
        self.saehd_path = Path(str(dfl_path / "saehd-models"))
        self.videos_path = Path(str(dfl_path / "videos"))
        self.videos_frames_path = Path(str(dfl_path / "videos-frames"))
        self.tmp_path = Path(str(dfl_path / "tmp"))
        self.dflab_repo = Path(str(dfl_path / "dflab"))
        self.dflive_repo = Path(str(dfl_path / "dflive"))

        if hasattr(opts, "opts.deepfacelab_dfl_path"):
            self.dfl_path = Path(opts.deepfacelab_dfl_path)

        if hasattr(opts, "deepfacelab_workspaces_path"):
            self.workspaces_path = Path(opts.deepfacelab_workspaces_path)

        if hasattr(opts, "deepfacelab_pak_path"):
            self.pak_path = Path(opts.deepfacelab_pak_path)

        if hasattr(opts, "deepfacelab_xseg_path"):
            self.xseg_path = Path(opts.deepfacelab_xseg_path)

        if hasattr(opts, "deepfacelab_saehd_path"):
            self.saehd_path = Path(opts.deepfacelab_saehd_path)

        if hasattr(opts, "deepfacelab_videos_path"):
            self.videos_path = Path(opts.deepfacelab_videos_path)

        if hasattr(opts, "deepfacelab_videos_frames_path"):
            self.videos_frames_path = Path(opts.deepfacelab_videos_frames_path)

        if hasattr(opts, "deepfacelab_tmp_path"):
            self.tmp_path = Path(opts.deepfacelab_tmp_path)

        if hasattr(opts, "deepfacelab_dflab_repo_path"):
            self.dflab_repo = Path(opts.deepfacelab_dflab_repo_path)

        if hasattr(opts, "deepfacelab_dflive_repo_path"):
            self.dflive_repo = Path(opts.deepfacelab_dflive_repo_path)

        # Create dirs if not existing
        for p in [self.dfl_path, self.workspaces_path, self.pak_path, self.xseg_path, self.saehd_path, self.videos_path, self.videos_frames_path, self.tmp_path]:
            DflFiles.create_folder(p)


    # Getters start here

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

    # Lists start here

    def get_dfl_list(self, include_downloadable=True):
        dir_files = DflFiles.get_files_from_dir(self.dfl_path, [".dfm"], True)
        dir_files_one = DflFiles.get_files_from_dir(self.dfl_path, [".dfm"])
        if include_downloadable:
            downloable_files = self.get_downloadable_models(dir_files_one)
            tmp_files = []
            for f in downloable_files:
                tmp_files.append([f[0] + " (To Download)", f[1]])
            return dir_files + tmp_files
        return dir_files

    def get_downloadable_models(self, available_models):
        from scripts.command import get_downloadable_models
        return get_downloadable_models(self.dfl_path, available_models)

    def get_pak_list(self):
        return DflFiles.get_files_from_dir(self.pak_path, [".pak"])

    def get_videos_list(self):
        return DflFiles.get_files_from_dir(self.videos_path, [".mp4", ".mkv"])

    def get_videos_list_full_path(self):
        return list(self.videos_path + "/" + v for v in self.get_videos_list())

    def get_workspaces_list(self):
        return DflFiles.get_folder_names_in_dir(self.workspaces_path)

    def get_saehd_models_list(self):
        return DflFiles.get_folder_names_in_dir(self.saehd_path)

    def get_xseg_models_list(self):
        return DflFiles.get_folder_names_in_dir(self.xseg_path)