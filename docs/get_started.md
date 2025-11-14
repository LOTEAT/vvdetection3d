# Get Started

## Prerequisites

In this section, we demonstrate how to prepare an environment with PyTorch.

VVDetection3D works on Linux, Windows (experimental support) and macOS. It requires Python 3.7+, CUDA 10.0+, and PyTorch 1.8+.

```{note}
If you are experienced with PyTorch and have already installed it, just skip this part and jump to the [next section](#installation). Otherwise, you can follow these steps for the preparation.
```

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name vvdet3d python=3.8 -y
conda activate vvdet3d
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```shell
conda install pytorch torchvision -c pytorch
```

On CPU platforms:

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

## Installation

We recommend that users follow our best practices to install VVDetection3D. However, the whole process is highly customizable. See [Customize Installation](#customize-installation) section for more information.

### Best Practices

**Step 0.** Install [MMEngine](https://github.com/open-mmlab/mmengine), [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc4'
mim install 'mmdet>=3.0.0'
```

**Note**: In MMCV-v2.x, `mmcv-full` is renamed to `mmcv`, if you want to install `mmcv` without CUDA ops, you can use `mim install "mmcv-lite>=2.0.0rc4"` to install the lite version.

**Step 1.** Install VVDetection3D.

Install it from source:

```shell
git clone https://github.com/LOTEAT/vvdetection3d.git
# "-b dev-1.x" means checkout to the `dev-1.x` branch.
cd vvdetection3d
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in edtiable mode,
# thus any local modifications made to the code will take effect without reinstallation.

Note:

1. If you would like to use `opencv-python-headless` instead of `opencv-python`,
   you can install it before installing MMCV.

2. Some dependencies are optional. Simply running `pip install -v -e .` will only install the minimum runtime requirements. To use optional dependencies like `albumentations` and `imagecorruptions` either install them manually with `pip install -r requirements/optional.txt` or specify desired extras when calling `pip` (e.g. `pip install -v -e .[optional]`). Valid keys for the extras field are: `all`, `tests`, `build`, and `optional`.

   We have supported `spconv 2.0`. If the user has installed `spconv 2.0`, the code will use `spconv 2.0` first, which will take up less GPU memory than using the default `mmcv spconv`. Users can use the following commands to install `spconv 2.0`:

   ```shell
   pip install cumm-cuxxx
   pip install spconv-cuxxx
   ```

   Where `xxx` is the CUDA version in the environment.

   For example, using CUDA 10.2, the command will be `pip install cumm-cu102 && pip install spconv-cu102`.

   Supported CUDA versions include 10.2, 11.1, 11.3, and 11.4. Users can also install it by building from the source. For more details please refer to [spconv v2.x](https://github.com/traveller59/spconv).

   We also support `Minkowski Engine` as a sparse convolution backend. If necessary please follow original [installation guide](https://github.com/NVIDIA/MinkowskiEngine#installation) or use `pip` to install it:

   ```shell
   conda install openblas-devel -c anaconda
   export CPLUS_INCLUDE_PATH=CPLUS_INCLUDE_PATH:${YOUR_CONDA_ENVS_DIR}/include
   # replace ${YOUR_CONDA_ENVS_DIR} to your anaconda environment path e.g. `/home/username/anaconda3/envs/openmmlab`.
   pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=/opt/conda/include" --install-option="--blas=openblas"
   ```

   We also support `Torchsparse` as a sparse convolution backend. If necessary please follow original [installation guide](https://github.com/mit-han-lab/torchsparse#installation) or use `pip` to install it:

   ```shell
   sudo apt-get install libsparsehash-dev
   pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
   ```

   or omit sudo install by following command:

   ```shell
   conda install -c bioconda sparsehash
   export CPLUS_INCLUDE_PATH=CPLUS_INCLUDE_PATH:${YOUR_CONDA_ENVS_DIR}/include
   # replace ${YOUR_CONDA_ENVS_DIR} to your anaconda environment path e.g. `/home/username/anaconda3/envs/openmmlab`.
   pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
   ```