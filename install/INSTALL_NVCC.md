# Installation of the CUDA toolkit

To install the CUDA toolkit (NVCC _etc._) you can either install the conda environment, which handles this for you, or do it manually (if using pip/uv). [You can also follow this link.](https://developer.nvidia.com/cuda-downloads)

## Ubuntu

Note

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-0
```

## Windows

Go to [this link](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network), download the exe, double click to run & follow the instructions.

## Flash attn

You may then need to set the CUDA*HOME environment variable when installing flash-attn \_i.e*

```bash
MAX_JOBS=4 CUDA_HOME=path/you/installed/cuda/to pip install --no-build-isolation flash-attn
```
