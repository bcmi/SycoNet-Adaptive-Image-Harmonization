# SycoNet: Domain Adaptive Image Harmonization

We release the SycoNet inference code used in our paper "Deep Image Harmonization with Learnable Augmentation", ICCV2023. SycoNet can generate multiple plausible synthetic composite images based on a real image and a foreground mask, which is useful to construct pairs of synthetic composite images and real images for harmonization. 

<div align="center">
	<img src="figures/flowchart..jpg" alt="SycoNet" width="800">
</div>

# Setup

Clone the repository:
```
git clone git@github.com:bcmi/SycoNet-Adaptive-Image-Harmonization.git
```
Install Anaconda and create a virtual environment:
```
conda create -n syconet python=3.6
conda activate syconet
```
Install PyTorch:
```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```
Install necessary packages:
```
pip install -r requirements.txt
```
Build Trilinear:
```
cd trilinear_cpp
sh setup.sh
```
Modify `CUDA_HOME` as your own path in `setup.sh`. You can refer to [this repository](https://github.com/HuiZeng/Image-Adaptive-3DLUT) for more solutions.

# Inference

Download pre-trained SycoNet `pretrained_net_Er.pth` and 3D LUTs `pretrained_net_LUTs.pth` from [Baidu Cloud](https://pan.baidu.com/s/1wIWxb37yIVccxB0kM-FnnQ) (access code:o4rt). Put them in the folder `checkpoints\syco`. 

Modify `real` and `mask` in `demo_test.sh` as your own real image path and foreground mask path respectively. Modify  `augment_num` as your expected number of generated composite images per pair of real image and foreground mask. Then, run the following command:
```
sh demo_test.sh
```
Our SycoNet could generate composite images for the input real image and foreground mask in the folder `results\syco\test_pretrained`.

