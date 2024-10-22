import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple, Optional
from packaging import version
import importlib.metadata

import launch
from launch import is_installed, run, run_pip

# Determine whether to skip installation based on command-line arguments
try:
    skip_install = getattr(launch.args, "skip_install", False)
except AttributeError:
    skip_install = getattr(launch, "skip_install", False)

python = sys.executable

def comparable_version(version_str: str) -> Tuple[int, ...]:
    """Convert a version string into a tuple of integers for comparison."""
    return tuple(map(int, version_str.split(".")))

def get_installed_version(package: str) -> Optional[str]:
    """Retrieve the installed version of a package, if available."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None

def install_uddetailer():
    """Install and manage dependencies for the 'uddetailer' component."""
    if not is_installed("mim"):
        run_pip("install -U openmim", desc="Installing openmim")

    # Ensure minimum requirements are met
    if not is_installed("mediapipe"):
        run_pip('install protobuf>=3.20', desc="Installing protobuf")
        run_pip('install mediapipe>=0.10.3', desc="Installing mediapipe")

    torch_version = get_installed_version("torch")
    legacy = torch_version and comparable_version(torch_version)[0] < 2

    # Check versions and manage installations for mmdet and mmcv
    mmdet_version = get_installed_version("mmdet")
    mmdet_v3 = mmdet_version and version.parse(mmdet_version) >= version.parse("3.0.0")

    if not is_installed("mmdet") or (legacy and mmdet_v3) or (not legacy and not mmdet_v3):
        if legacy and mmdet_v3:
            print("Uninstalling mmdet and mmengine...")
            run(f'"{python}" -m pip uninstall -y mmdet mmengine', live=True)
        run(f'"{python}" -m mim install mmcv-full', desc="Installing mmcv-full")
        run_pip("install mmdet==2.28.2", desc="Installing mmdet")
    else:
        if not mmdet_v3:
            print("Uninstalling mmdet, mmcv, and mmcv-full...")
            run(f'"{python}" -m pip uninstall -y mmdet mmcv mmcv-full', live=True)
        print("Installing mmcv, mmdet, and mmengine...")
        if not is_installed("mmengine"):
            run_pip("install mmengine==0.8.5", desc="Installing mmengine")
        if version.parse(torch_version) >= version.parse("2.1.0"):
            run(f'"{python}" -m mim install mmcv~=2.1.0', desc="Installing mmcv 2.1.0")
        else:
            run(f'"{python}" -m mim install mmcv~=2.0.0', desc="Installing mmcv")
        run(f'"{python}" -m mim install -U mmdet>=3.0.0', desc="Installing mmdet")
        run_pip("install mmdet>=3", desc="Installing mmdet")

    # Verify mmcv and mmengine versions
    mmcv_version = get_installed_version("mmcv")
    if mmcv_version and version.parse(mmcv_version) >= version.parse("2.0.1"):
        print(f"Your mmcv version {mmcv_version} may not work with mmyolo.")
        print("Consider fixing the version restriction manually.")

    mmengine_version = get_installed_version("mmengine")
    if mmengine_version:
        if version.parse(mmengine_version) >= version.parse("0.9.0"):
            print(f"Your mmengine version {mmengine_version} may not work on Windows.")
            print("Install mmengine 0.8.5 manually or use an updated version of bitsandbytes.")
        else:
            print(f"Your mmengine version is {mmengine_version}")

    if not legacy and not is_installed("mmyolo"):
        run(f'"{python}" -m pip install mmyolo', desc="Installing mmyolo")

    # Install additional requirements from requirements.txt
    req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
    if os.path.exists(req_file):
        mainpackage = 'sd-deepface-1111'
        with open(req_file) as file:
            for package in file:
                package = package.strip()
                try:
                    if '==' in package:
                        package_name, package_version = package.split('==')
                        installed_version = get_installed_version(package_name)
                        if installed_version != package_version:
                            run_pip(f"install -U {package}", desc=f"{mainpackage} requirement: updating {package_name} to {package_version}")
                    elif '>=' in package:
                        package_name, package_version = package.split('>=')
                        installed_version = get_installed_version(package_name)
                        if not installed_version or comparable_version(installed_version) < comparable_version(package_version):
                            run_pip(f"install -U {package}", desc=f"{mainpackage} requirement: updating {package_name} to {package_version}")
                    elif not is_installed(package):
                        run_pip(f"install {package}", desc=f"{mainpackage} requirement: {package}")
                except Exception as e:
                    print(f"Error installing {package}: {e}")

def install():
    """Install essential packages for DeepFaceLab and related tools."""
    packages = {
        "tqdm": "requirements for DeepFaceLab - tqdm",
        "numpy": "requirements for DeepFaceLab - numpy",
        "numexpr": "requirements for DeepFaceLab - numexpr",
        "h5py": "requirements for DeepFaceLab - h5py",
        "opencv-python": "requirements for DeepFaceLab - opencv-python",
        "opencv-contrib-python": "requirements for DeepFaceLab - opencv-contrib-python",
        "ffmpeg-python": "requirements for DeepFaceLab - ffmpeg-python",
        "scikit-image": "requirements for DeepFaceLab - scikit-image",
        "scipy": "requirements for DeepFaceLab - scipy",
        "colorama": "requirements for DeepFaceLab - colorama",
        "tensorflow": "requirements for DeepFaceLab - tensorflow",
        "pyqt5": "requirements for DeepFaceLab - pyqt5",
        "tf2onnx": "requirements for DeepFaceLab - tf2onnx",
        "onnxruntime": "requirements for DeepFaceLab - onnxruntime",
        "onnxruntime-gpu": "requirements for DeepFaceLab - onnxruntime-gpu==1.12.1",
        "protobuf": "requirements for DeepFaceLab - protobuf==3.20.3",
    }

    for package, desc in packages.items():
        if not is_installed(package) or (package == "onnxruntime-gpu" and get_installed_version(package) != '1.12.1') or (package == "protobuf" and get_installed_version(package) != '3.20.3'):
            version_specifier = "" if package != "onnxruntime-gpu" and package != "protobuf" else "==1.12.1" if package == "onnxruntime-gpu" else "==3.20.3"
            run_pip(f"install {package}{version_specifier}", desc=desc)

def checkout_git_commit(repo_name: str, commit: str, output_folder: str):
    """Clone a GitHub repository and check out a specific commit."""
    if not os.path.isdir(output_folder):
        subprocess.run(['git', 'clone', f'https://github.com/{repo_name}.git', output_folder])

    os.chdir(output_folder)
    subprocess.run(['git', 'checkout', commit])

# Main script execution
if not skip_install:
    install()
    install_uddetailer()

script_path = Path(os.path.dirname(os.path.abspath(__file__))) / "repo/"
checkout_git_commit('idinkov/DeepFaceLive', 'af8396925cccc1d3f02867e16b8929060c3ebc5f', str(script_path / "dflive"))
