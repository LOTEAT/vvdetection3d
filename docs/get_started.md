# Get Started

## Environment

The recommended environment is Linux with Python 3.8+ and a CUDA-enabled
PyTorch installation.

The current tested PyTorch environment is:

```text
torch==1.13.1+cu117
torchvision==0.14.1+cu117
torchaudio==0.13.1+cu117
```

Create and activate a clean environment:

```shell
conda create -n vvdet3d python=3.8 -y
conda activate vvdet3d
```

Install PyTorch first by following the official selector at
<https://pytorch.org/get-started/locally/>.

## Installation

Install the OpenMMLab runtime dependencies:

```shell
pip install -U openmim
mim install "mmengine==0.10.7"
mim install "mmcv==2.0.0rc4"
mim install "mmdet==3.3.0"
```

Install the Python requirements used by this repository:

```shell
pip install -r requirements/runtime.txt
```

Install the project in editable mode:

```shell
pip install -v -e .
```

Optional dependency groups are available through:

```shell
pip install -r requirements/optional.txt
pip install -r requirements/tests.txt
```

## Quick Check

After installation, make sure the package can be imported:

```shell
python -c "import vvdet3d; print(vvdet3d.__version__)"
```

## Example Workflow

Prepare the dataset by following [VVSim Dataset](./vvsim.md).

Current limitation: training is only supported on a single GPU with
`batch_size=1`.

Training and evaluation examples are documented in [VVSim Dataset](./vvsim.md).
