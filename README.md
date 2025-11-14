# VVDetection3d

## Introduction

VVDetection3D is a collaborative 3D object detection framework built on PyTorch and MMDetection3D. It focuses on multi-agent perception, enabling vehicles or UAVs to share and fuse information for more accurate and robust 3D detection in challenging environments. The framework adopts a modular design with reusable data pipelines, configuration-driven training, and plug-and-play model components, making it easy to create and train your own model. By leveraging the mature ecosystem of MMDetection3D, VVDetection3D provides familiar tooling, ~~strong baselines~~ (is coming), and a flexible extension space for rapid research and deployment.


## Installation

Please refer to [Installation](./docs/get_started.md) for installation instructions.


## Get Started

For detailed user guides and advanced guides, please refer to [mmdetection3d documentation](https://mmdetection3d.readthedocs.io/en/latest/) (very similar):

<details>
<summary>User Guides</summary>

- [Train & Test](https://mmdetection3d.readthedocs.io/en/latest/user_guides/index.html#train-test)
  - [Learn about Configs](https://mmdetection3d.readthedocs.io/en/latest/user_guides/config.html)
  - [Coordinate System](https://mmdetection3d.readthedocs.io/en/latest/user_guides/coord_sys_tutorial.html)
  - [Dataset Preparation](https://mmdetection3d.readthedocs.io/en/latest/user_guides/dataset_prepare.html)
  - [Customize Data Pipelines](https://mmdetection3d.readthedocs.io/en/latest/user_guides/data_pipeline.html)
  - [Test and Train on Standard Datasets](https://mmdetection3d.readthedocs.io/en/latest/user_guides/train_test.html)
  - [Inference](https://mmdetection3d.readthedocs.io/en/latest/user_guides/inference.html)
  - [Train with Customized Datasets](https://mmdetection3d.readthedocs.io/en/latest/user_guides/new_data_model.html)
- [Useful Tools](https://mmdetection3d.readthedocs.io/en/latest/user_guides/index.html#useful-tools)

</details>

<details>
<summary>Advanced Guides</summary>

- Datasets
  - [VVSim Dataset](./docs/vvsim.md)
 
</details>


## Overview of Benchmark and Model Zoo


## Acknowledgement
VVDetection3D is deeply indebted to the open-source communities behind MMDetection3D and OpenCOOD. We sincerely thank all maintainers and contributors for their foundational designs, high-quality implementations, and generous community support. Their work has significantly accelerated our research and enabled robust, modular 3D detection and collaborative perception capabilities in this project.


## License
This project is released under the [Apache 2.0 license](./LICENSE).
