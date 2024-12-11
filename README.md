# SMamba: Sparse Mamba for Event-based Object Detection

This is the official Pytorch implementation of the AAAI2025 paper [SMamba: Sparse Mamba for Event-based Object Detection](https://ieeexplore.ieee.org/document/10682110).

## Conda Installation 
Step1: Same as [RVT](https://github.com/uzh-rpg/RVT)

```Bash
conda create -y -n sfnet python=3.9 pip
conda activate sfnet
conda config --set channel_priority flexible

CUDA_VERSION=11.8

conda install -y h5py=3.8.0 blosc-hdf5-plugin=1.0.0
hydra-core=1.3.2 einops=0.6.0 torchdata=0.6.0 tqdm numba
pytorch=2.0.0 torchvision=0.15.0 pytorch-cuda=$CUDA_VERSION
-c pytorch -c nvidia -c conda-forge

python -m pip install pytorch-lightning==1.8.6 wandb==0.14.0
pandas==1.5.3 plotly==5.13.1 opencv-python==4.6.0.66 tabulate==0.9.0
pycocotools==2.0.6 bbox-visualizer==0.1.0 StrEnum=0.4.10
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
Detectron2 is not strictly required but speeds up the evaluation.

Step2:

```Bash
cd kernels/selective_scan && pip install .
```
## Required Data
To evaluate or train SMamba you will need to download the required preprocessed datasets:

<table><tbody>
<th valign="bottom"></th>
<th valign="bottom">1 Mpx</th>
<th valign="bottom">Gen1</th>
  <th valign="bottom">eTram</th>
<tr><td align="left">pre-processed dataset</td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/datasets/preprocessed/gen4.tar">download</a></td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/datasets/preprocessed/gen1.tar">download</a></td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/datasets/preprocessed/gen1.tar">download</a></td>
</tr>
<tr><td align="left">crc32</td>
<td align="center"><tt>c5ec7c38</tt></td>
<td align="center"><tt>5acab6f3</tt></td>
<td align="center"><tt>5acab6f3</tt></td>
</tr>
</tbody></table>

## Code Acknowledgments
This project has used code from the following projects:
- [RVT](https://github.com/uzh-rpg/RVT) for the RVT architecture implementation in Pytorch
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) for the detection PAFPN/head
