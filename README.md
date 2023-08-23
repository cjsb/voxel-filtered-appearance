# Voxel-based Representations for Improved Filtered Appearance
[[Project page]](https://cjsb.github.io/hpg2023/)

Repository for **"Voxel-based Representations for Improved Filtered Appearance"**, published at **HPG 2023** by Caio Brito, Pierre Poulin, Veronica Teichrieb

## Installation

To use our code in your projects you will need to add [CUDA](https://developer.nvidia.com/cuda-toolkit), [EIGEN](https://eigen.tuxfamily.org/index.php?title=Main_Page) and [pcg32](https://github.com/wjakob/pcg32) to your project.

After that, include all '.h' files to your project. The representations (Virtual Mesh and Subgrig of Opacities) can be found in 'representations.h' file.

## Usage

See 'main.cpp' contains an example on how to use our code to build the representation and how to use it in shading and to compute occlusion.

## Citations
Please cite our paper if this code contributes to an academic publication using the following bibtex reference:

```bib
@inproceedings {10.2312:hpg.20231132,
booktitle = {High-Performance Graphics - Symposium Papers},
editor = {Bikker, Jacco and Gribble, Christiaan},
title = {{Voxel-based Representations for Improved Filtered Appearance}},
author = {Brito, Caio José Dos Santos and Poulin, Pierre and Teichrieb, Veronica},
year = {2023},
publisher = {The Eurographics Association},
ISSN = {2079-8687},
ISBN = {978-3-03868-229-5},
DOI = {10.2312/hpg.20231132}
}
```