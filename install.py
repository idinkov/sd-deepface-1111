import launch
import os
import subprocess

from pathlib import Path
from modules.shared import opts, OptionInfo
from modules import shared, paths, script_callbacks

if not launch.is_installed("tqdm"):
    launch.run_pip("install tqdm", "requirements for DeepFaceLab - tqdm")

if not launch.is_installed("numpy"):
    launch.run_pip("install numpy", "requirements for DeepFaceLab - numpy")

if not launch.is_installed("numexpr"):
    launch.run_pip("install numexpr", "requirements for DeepFaceLab - numexpr")

if not launch.is_installed("h5py"):
    launch.run_pip("install h5py==2.10.0", "requirements for DeepFaceLab - h5py")

if not launch.is_installed("opencv-python"):
    launch.run_pip("install opencv-python", "requirements for DeepFaceLab - opencv-python")

if not launch.is_installed("opencv-contrib-python"):
    launch.run_pip("install opencv-contrib-python", "requirements for DeepFaceLab - opencv-contrib-python")

if not launch.is_installed("ffmpeg-python"):
    launch.run_pip("install ffmpeg-python", "requirements for DeepFaceLab - ffmpeg-python")

if not launch.is_installed("scikit-image"):
    launch.run_pip("install scikit-image", "requirements for DeepFaceLab - scikit-image")

if not launch.is_installed("scipy"):
    launch.run_pip("install scipy", "requirements for DeepFaceLab - scipy")

if not launch.is_installed("colorama"):
    launch.run_pip("install colorama", "requirements for DeepFaceLab - colorama")

if not launch.is_installed("tensorflow"):
    launch.run_pip("install tensorflow", "requirements for DeepFaceLab - tensorflow")

if not launch.is_installed("pyqt5"):
    launch.run_pip("install pyqt5", "requirements for DeepFaceLab - pyqt5")

if not launch.is_installed("tf2onnx"):
    launch.run_pip("install tf2onnx", "requirements for DeepFaceLab - tf2onnx")

if not launch.is_installed("onnxruntime"):
    launch.run_pip("install onnxruntime", "requirements for DeepFaceLab - onnxruntime")

# if not launch.is_installed("protobuf"):
launch.run_pip("install protobuf==3.20", "requirements for DeepFaceLab - protobuf==3.20")

script_path = Path(paths.script_path) / "extensions/deepfacelab/repo/"
dflab_path = str(script_path / "dflab")
dflive_path = str(script_path / "dflive")

def checkout_git_commit(repo_name, commit, output_folder):
    print("checkout" + repo_name + "["+commit+"]" + output_folder)
    # Clone the repository if it doesn't exist
    if not os.path.isdir(output_folder):
        subprocess.run(['git', 'clone', f'https://github.com/{repo_name}.git', output_folder])

    # Change directory to the repository
    os.chdir(output_folder)

    # Checkout the specified commit
    subprocess.run(['git', 'checkout', commit])

#checkout_git_commit('iperov/DeepFaceLab', '9ef04b2207ba9527a9991094de14b79d2ed8188a', dflab_path)
checkout_git_commit('idinkov/DeepFaceLive', 'af8396925cccc1d3f02867e16b8929060c3ebc5f', dflive_path)

