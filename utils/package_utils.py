import os
import subprocess

def install_torch():
    # install pip
    os.system("python -m pip install --upgrade pip")
    # install torch
    try:
        # if gpu is available
        subprocess.check_output('nvidia-smi')
        print('Nvidia GPU detected!')
        os.system("pip3 install torch==1.13.1")
        # the following command is from 
        # https://stackoverflow.com/questions/74394695/how-does-one-fix-when-torch-cant-find-cuda-error-version-libcublaslt-so-11-no
        os.system("pip3 uninstall nvidia_cublas_cu11 --yes")
    except Exception: # this command not being found can raise quite a few different errors depending on the configuration
        # if only cpu is available
        print('No Nvidia GPU in system!')
        os.system("pip3 install torch==1.13.1 --extra-index-url https://download.pytorch.org/whl/cpu")
        os.system("pip3 uninstall nvidia_cublas_cu11 --yes")


def install_faiss():
    # install pip
    os.system("python -m pip install --upgrade pip")
    # install faiss-cpu
    os.system("pip3 install faiss-cpu")


if __name__ == "__main__":
    install_torch()
    install_faiss()

