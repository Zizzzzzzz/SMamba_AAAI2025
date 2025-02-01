# SMamba: Sparse Mamba for Event-based Object Detection

This is the official Pytorch implementation of the AAAI2025 paper [SMamba: Sparse Mamba for Event-based Object Detection].

## Conda Installation 
Step1: Same as [RVT](https://github.com/uzh-rpg/RVT)

```Bash
conda create -y -n smamba python=3.9 pip
conda activate smamba
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
<td align="center"><a href="https://docs.google.com/forms/d/e/1FAIpQLSfH2LI5oqWWfose-pBC3dsbaAMvRQuv0BI93njV_5wQjYx83w/viewform">download</a></td>
</tr>
<tr><td align="left">crc32</td>
<td align="center"><tt>c5ec7c38</tt></td>
<td align="center"><tt>5acab6f3</tt></td>
<td align="center"><tt>-</tt></td>
</tr>
</tbody></table>

## Pre-trained Checkpoints
<table><tbody>
<th valign="bottom"></th>
<th valign="bottom">1 Mpx</th>
<th valign="bottom">Gen1</th>
  <th valign="bottom">eTram</th>
<tr><td align="left">links</td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/datasets/preprocessed/gen4.tar">download</a></td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/datasets/preprocessed/gen1.tar">download</a></td>
<td align="center"><a href="https://docs.google.com/forms/d/e/1FAIpQLSfH2LI5oqWWfose-pBC3dsbaAMvRQuv0BI93njV_5wQjYx83w/viewform">download</a></td>
</tbody></table>

## Evaluation
- Set `DATA_DIR` as the path to either the eTram, 1 Mpx or Gen1 dataset directory
- Set `CKPT_PATH` to the path of the *correct* checkpoint matching the choice of the model and dataset.
- Set
  - `USE_TEST=1` to evaluate on the test set, or
  - `USE_TEST=0` to evaluate on the validation set
- Set `GPU_ID` to the PCI BUS ID of the GPU that you want to use. e.g. `GPU_ID=0`.
  Only a single GPU is supported for evaluation
  
### eTram
```Bash
python validation.py dataset=etram dataset.path=${DATA_DIR} checkpoint=${CKPT_PATH} \
use_test_set=${USE_TEST} hardware.gpus=${GPU_ID} +experiment/etram="base.yaml" \
batch_size.eval=4 model.postprocess.confidence_threshold=0.01
```
### 1 Mpx
```Bash
python validation.py dataset=gen4 dataset.path=${DATA_DIR} checkpoint=${CKPT_PATH} \
use_test_set=${USE_TEST} hardware.gpus=${GPU_ID} +experiment/gen4="base.yaml" \
batch_size.eval=8 model.postprocess.confidence_threshold=0.01
```
### Gen1
```Bash
python validation.py dataset=gen1 dataset.path=${DATA_DIR} checkpoint=${CKPT_PATH} \
use_test_set=${USE_TEST} hardware.gpus=${GPU_ID} +experiment/gen1="base.yaml" \
batch_size.eval=8 model.postprocess.confidence_threshold=0.01
```
## Training
- Set `DATA_DIR` as the path to either the eTram, 1 Mpx or Gen1 dataset directory
- Set `GPU_IDS` to the PCI BUS IDs of the GPUs that you want to use. e.g. `GPU_IDS=[0,1]` for using GPU 0 and 1.
  **Using a list of IDS will enable single-node multi-GPU training.**
  Pay attention to the batch size which is defined per GPU:
- Set `BATCH_SIZE_PER_GPU` such that the effective batch size is matching the parameters below.
  The **effective batch size** is (batch size per gpu)*(number of GPUs).
- If you would like to change the effective batch size, we found the following learning rate scaling to work well for 
all models on both datasets:
  
  `lr = 2e-4 * sqrt(effective_batch_size/8)`.
- The training code uses [W&B](https://wandb.ai/) for logging during the training.
Hence, we assume that you have a W&B account.

### eTram
```Bash
GPU_IDS=[0,1]
BATCH_SIZE_PER_GPU=4
TRAIN_WORKERS_PER_GPU=2
EVAL_WORKERS_PER_GPU=2
python train.py model=rnndet dataset=etram dataset.path=${DATA_DIR} wandb.project_name=SMamba \
wandb.group_name=etram +experiment/etram="base.yaml" hardware.gpus=${GPU_IDS} \
batch_size.train=${BATCH_SIZE_PER_GPU} batch_size.eval=${BATCH_SIZE_PER_GPU} \
hardware.num_workers.train=${TRAIN_WORKERS_PER_GPU} hardware.num_workers.eval=${EVAL_WORKERS_PER_GPU}
```
### 1 Mpx
```Bash
GPU_IDS=[0,1]
BATCH_SIZE_PER_GPU=8
TRAIN_WORKERS_PER_GPU=2
EVAL_WORKERS_PER_GPU=2
python train.py model=rnndet dataset=gen4 dataset.path=${DATA_DIR} wandb.project_name=SMamba \
wandb.group_name=1mpx +experiment/gen4="base.yaml" hardware.gpus=${GPU_IDS} \
batch_size.train=${BATCH_SIZE_PER_GPU} batch_size.eval=${BATCH_SIZE_PER_GPU} \
hardware.num_workers.train=${TRAIN_WORKERS_PER_GPU} hardware.num_workers.eval=${EVAL_WORKERS_PER_GPU}
```
### GEN1
```Bash
GPU_IDS=[0,1]
BATCH_SIZE_PER_GPU=8
TRAIN_WORKERS_PER_GPU=2
EVAL_WORKERS_PER_GPU=2
python train.py model=rnndet dataset=gen1 dataset.path=${DATA_DIR} wandb.project_name=SMamba \
wandb.group_name=gen1 +experiment/gen1="base.yaml" hardware.gpus=${GPU_IDS} \
batch_size.train=${BATCH_SIZE_PER_GPU} batch_size.eval=${BATCH_SIZE_PER_GPU} \
hardware.num_workers.train=${TRAIN_WORKERS_PER_GPU} hardware.num_workers.eval=${EVAL_WORKERS_PER_GPU}
```

## Code Acknowledgments
This project has used code from the following projects:
- [RVT](https://github.com/uzh-rpg/RVT) for the RVT architecture implementation in Pytorch
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) for the detection PAFPN/head
- [VMamba](https://github.com/MzeroMiko/VMamba) for the VSS block
