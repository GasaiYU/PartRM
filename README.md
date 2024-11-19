# PartRM: Modeling Part-Level Dynamics with Large 4D Reconstruction Model

This repository is an official implementation for:

**PartRM: Modeling Part-Level Dynamics with Large 4D Reconstruction Model**

![Teaser](./images/teaser.png)

## Introduction
As interest grows in world models that predict future states from current observations and actions, accurately modeling part-level dynamics has become increasingly relevant for various applications. Existing approaches, such as Puppet-Master, rely on fine-tuning large-scale pre-trained video diffusion models, which are impractical for real-world use due to the limitations of 2D video representation and slow processing times. To overcome these challenges, we present PartRM, a novel 4D reconstruction framework that simultaneously models appearance, geometry, and part-level motion from multi-view images of a static object. PartRM builds upon large 3D Gaussian reconstruction models, leveraging their extensive knowledge of appearance and geometry in static objects. To address data scarcity in 4D, we introduce the PartDrag-4D dataset, providing multi-view observations of part-level dynamics across over 20,000 states. We enhance the model’s understanding of interaction conditions with a multi-scale drag embedding module that captures dynamics at varying granularities. To prevent catastrophic forgetting during fine-tuning, we implement a two-stage training process that focuses sequentially on motion and appearance learning. Experimental results show that PartRM establishes a new state-of-the-art in part-level motion learning and can be applied in manipulation tasks in robotics. Our code, data, and models will be made publicly available to facilitate future research.

## Environment Setup
Use `conda` to create a new virtual enviroment
```
conda env create -f environment.yaml
conda activate partrm
```
Also with gaussian splatting renderer
```
# a modified gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization
```

## PartDrag-4D Dataset
You need to first get PartNet-Mobility dataset and put it in the `PartDrag4D/data` dir of this repo.
Then
```
cd PartDrag4D
```
For mesh preprocess and mesh animating:
```
cd preprocess
python process_data_textured_uv.py
python animated_data.py
```
For rendering
First download blender:
```
cd ../rendering/blender
wget https://download.blender.org/release/Blender3.5/blender-3.5.0-linux-x64.tar.xz
```

Then
```
cd ..
bash render.sh
```
You can modify `num_gpus` and `CUDA_VISIBLE_DEVICES` to adjust the degree of parallelism.
