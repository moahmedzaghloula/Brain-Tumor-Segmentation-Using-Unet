import os
import subprocess
import sys
import venv

# Define constants
ENV_NAME = "brain_tumor_env"
PYTHON_VERSION = "3.9"
PROJECT_DIR = os.path.expanduser("~/brain_tumor_segmentation_update")
VENV_PATH = os.path.join(PROJECT_DIR, ENV_NAME)
REQUIREMENTS = [
    "opencv-python==4.10.0.84",
    "numpy==1.24.4",  # Compatible with tensorflow 2.15
    "pandas==2.0.3",
    "matplotlib==3.7.5",
    "scikit-image==0.21.0",
    "pillow==10.4.0",
    "nibabel==5.2.1",
    "tensorflow==2.15.0",  # Updated for better compatibility
    "keras==2.15.0",
    "scikit-learn==1.3.2"
]

def check_python_version():
    """Check if Python 3.9 is installed."""
    try:
        result = subprocess.run(["python3.9", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            return True
        return False
    except FileNotFoundError:
        return False

def create_virtual_env():
    """Create a virtual environment if it doesn't exist."""
    if not os.path.exists(VENV_PATH):
        print(f"Creating virtual environment: {ENV_NAME}")
        if check_python_version():
            subprocess.check_call(["python3.9", "-m", "venv", VENV_PATH])
        else:
            print("Error: Python 3.9 is not installed.")
            print("Install Python 3.9 using:")
            print("  sudo apt update")
            print("  sudo apt install python3.9 python3.9-venv python3.9-dev")
            print("Then rerun this script.")
            sys.exit(1)
    else:
        print(f"Virtual environment {ENV_NAME} already exists")

def get_pip_path():
    """Get the path to pip in the virtual environment."""
    pip_path = os.path.join(VENV_PATH, "bin", "pip")
    return pip_path

def install_dependencies():
    """Install dependencies in the virtual environment."""
    pip_path = get_pip_path()
    
    # Upgrade pip
    subprocess.check_call([pip_path, "install", "--upgrade", "pip"])
    
    # Install dependencies
    for package in REQUIREMENTS:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([pip_path, "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")
            print("Continuing with next package...")

def save_requirements():
    """Save dependencies to a requirements.txt file."""
    requirements_path = os.path.join(PROJECT_DIR, "requirements.txt")
    with open(requirements_path, "w") as f:
        for package in REQUIREMENTS:
            f.write(f"{package}\n")
    print(f"Saved dependencies to {requirements_path}")

def main():
    print("Setting up Python environment for brain tumor segmentation...")
    os.chdir(PROJECT_DIR)
    create_virtual_env()
    install_dependencies()
    save_requirements()
    print(f"Environment setup complete. Activate with:")
    print(f"source {VENV_PATH}/bin/activate")
    print("To verify, run: pip list")
    print("To test imports, run: python -c 'import cv2, numpy, pandas, matplotlib, skimage, PIL, nibabel, keras, tensorflow, sklearn'")

if __name__ == "__main__":
    main()